//! Shared circuit and helpers of the `gpu_hooks_*` tests. Each test file is
//! its own process, because hooks are installed once per process.
//!
//! The circuit is the one of `plonk_api.rs` (custom gates, copy constraints,
//! a lookup, two instances of the circuit in one proof), with a lookup table
//! that keeps the stock prover deterministic (see `values`).
#![allow(dead_code)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::op_ref)]

use std::marker::PhantomData;

use blake2b_simd::State;
use ff::Field;
use midnight_curves::{Bls12, Fq};
use midnight_proofs::{
    circuit::{Cell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk_with_k, prepare, Advice, Circuit, Column,
        ConstraintSystem, Constraints, Error, Fixed, ProvingKey, TableColumn,
    },
    poly::{
        commitment::Guard,
        kzg::{params::ParamsKZG, KZGCommitmentScheme},
        Rotation,
    },
    transcript::{CircuitTranscript, Transcript},
    utils::rational::Rational,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// Circuit size.
pub const K: u32 = 8;

/// The commitment scheme.
pub type Scheme = KZGCommitmentScheme<Bls12>;

/// BLAKE2b-256 of the stock proof (this branch, no hooks, feature `gpu` off,
/// 4 rayon threads), recorded from a run of `gpu_hooks_none` without the
/// feature.
pub const EXPECTED_PROOF_DIGEST: &str =
    "2e9a3d3a32e5d4b8d6cc0f107c7ca3f848ec38a5223c5e5f8f2b94e2fceb1e75";

#[derive(Clone)]
pub struct PlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    d: Column<Advice>,
    e: Column<Advice>,
    sa: Column<Fixed>,
    sb: Column<Fixed>,
    sc: Column<Fixed>,
    sm: Column<Fixed>,
    sp: Column<Fixed>,
    sl: TableColumn,
}

#[derive(Clone)]
pub struct MyCircuit<F: Field> {
    a: Value<F>,
    lookup_table: Vec<F>,
}

struct StandardPlonk<F: Field> {
    config: PlonkConfig,
    _marker: PhantomData<F>,
}

type Triple<FF> = (Rational<FF>, Rational<FF>, Rational<FF>);

impl<FF: Field> StandardPlonk<FF> {
    fn row(
        &self,
        layouter: &mut impl Layouter<FF>,
        name: &'static str,
        coeffs: [FF; 4],
        mut f: impl FnMut() -> Value<Triple<FF>>,
    ) -> Result<(Cell, Cell, Cell), Error> {
        layouter.assign_region(
            || name,
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                region.assign_advice(
                    || "lhs^4",
                    self.config.d,
                    0,
                    || value.unwrap().map(|v| v.0).square().square(),
                )?;
                let rhs = region.assign_advice(
                    || "rhs",
                    self.config.b,
                    0,
                    || value.unwrap().map(|v| v.1),
                )?;
                region.assign_advice(
                    || "rhs^4",
                    self.config.e,
                    0,
                    || value.unwrap().map(|v| v.1).square().square(),
                )?;
                let out = region.assign_advice(
                    || "out",
                    self.config.c,
                    0,
                    || value.unwrap().map(|v| v.2),
                )?;
                region.assign_fixed(|| "a", self.config.sa, 0, || Value::known(coeffs[0]))?;
                region.assign_fixed(|| "b", self.config.sb, 0, || Value::known(coeffs[1]))?;
                region.assign_fixed(|| "c", self.config.sc, 0, || Value::known(coeffs[2]))?;
                region.assign_fixed(|| "a * b", self.config.sm, 0, || Value::known(coeffs[3]))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }

    fn copy(&self, layouter: &mut impl Layouter<FF>, left: Cell, right: Cell) -> Result<(), Error> {
        layouter.assign_region(
            || "copy",
            |mut region| {
                region.constrain_equal(left, right)?;
                region.constrain_equal(left, right)
            },
        )
    }

    fn public_input(&self, layouter: &mut impl Layouter<FF>, v: FF) -> Result<Cell, Error> {
        layouter.assign_region(
            || "public_input",
            |mut region| {
                let value =
                    region.assign_advice(|| "value", self.config.a, 0, || Value::known(v))?;
                region.assign_fixed(|| "public", self.config.sp, 0, || Value::known(FF::ONE))?;
                Ok(value.cell())
            },
        )
    }

    fn lookup_table(&self, layouter: &mut impl Layouter<FF>, values: &[FF]) -> Result<(), Error> {
        layouter.assign_table(
            || "",
            |mut table| {
                for (index, &value) in values.iter().enumerate() {
                    table.assign_cell(
                        || "table col",
                        self.config.sl,
                        index,
                        || Value::known(value),
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    type Config = PlonkConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            a: Value::unknown(),
            lookup_table: self.lookup_table.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
        let e = meta.advice_column();
        let a = meta.advice_column();
        let b = meta.advice_column();
        let sf = meta.fixed_column();
        let c = meta.advice_column();
        let d = meta.advice_column();
        let p = meta.instance_column();

        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);

        let sm = meta.fixed_column();
        let sa = meta.fixed_column();
        let sb = meta.fixed_column();
        let sc = meta.fixed_column();
        let sp = meta.fixed_column();
        let sl = meta.lookup_table_column();

        meta.lookup("lookup", |meta| {
            let a_ = meta.query_any(a, Rotation::cur());
            vec![(a_, sl)]
        });

        meta.create_gate("Combined add-mult", |meta| {
            let d = meta.query_advice(d, Rotation::next());
            let a = meta.query_advice(a, Rotation::cur());
            let sf = meta.query_fixed(sf, Rotation::cur());
            let e = meta.query_advice(e, Rotation::prev());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let sa = meta.query_fixed(sa, Rotation::cur());
            let sb = meta.query_fixed(sb, Rotation::cur());
            let sc = meta.query_fixed(sc, Rotation::cur());
            let sm = meta.query_fixed(sm, Rotation::cur());
            Constraints::without_selector(vec![
                &a * sa + &b * sb + a * b * sm - (c * sc) + sf * (d * e),
            ])
        });

        meta.create_gate("Public input", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let p = meta.query_instance(p, Rotation::cur());
            let sp = meta.query_fixed(sp, Rotation::cur());
            Constraints::without_selector(vec![sp * (a - p)])
        });

        meta.enable_equality(sf);
        meta.enable_equality(e);
        meta.enable_equality(d);
        meta.enable_equality(p);
        meta.enable_equality(sm);
        meta.enable_equality(sa);
        meta.enable_equality(sb);
        meta.enable_equality(sc);
        meta.enable_equality(sp);

        PlonkConfig {
            a,
            b,
            c,
            d,
            e,
            sa,
            sb,
            sc,
            sm,
            sp,
            sl,
        }
    }

    fn synthesize(&self, config: PlonkConfig, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let cs = StandardPlonk::<F> {
            config,
            _marker: PhantomData,
        };
        let _ = cs.public_input(&mut layouter, F::ONE + F::ONE)?;
        for _ in 0..10 {
            let a: Value<Rational<_>> = self.a.into();
            let mut a_squared = Value::unknown();
            let (a0, _, c0) = cs.row(
                &mut layouter,
                "raw_multiply",
                [F::ZERO, F::ZERO, F::ONE, F::ONE],
                || {
                    a_squared = a.square();
                    a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
                },
            )?;
            let (a1, b1, _) = cs.row(
                &mut layouter,
                "raw_add",
                [F::ONE, F::ONE, F::ONE, F::ZERO],
                || {
                    let fin = a_squared + a;
                    a.zip(a_squared).zip(fin).map(|((a, a_squared), fin)| (a, a_squared, fin))
                },
            )?;
            cs.copy(&mut layouter, a0, a1)?;
            cs.copy(&mut layouter, b1, c0)?;
        }
        cs.lookup_table(&mut layouter, &self.lookup_table)?;
        Ok(())
    }
}

fn values() -> (Fq, Fq, Vec<Fq>) {
    use ff::WithSmallOrderMulGroup;
    let a = Fq::from(2834758237) * Fq::ZETA;
    let instance = Fq::ONE + Fq::ONE;
    // The stock lookup prover fills the permuted table with the leftover table
    // values in the iteration order of a `std` `HashMap` (random per process
    // and per map), so a seeded proof is reproducible only when the leftover
    // values are all equal. This table (its row-0 value is also the fill of
    // the unassigned rows) leaves only zeros over.
    (a, instance, vec![Fq::ZERO, instance, a])
}

/// Seeded parameters and the proving key.
///
/// Pins the rayon pool to 4 threads first: the stock vanishing argument
/// splits its random polynomial into one seeded RNG per pool thread, so a
/// seeded proof depends on the thread count.
pub fn setup() -> (ParamsKZG<Bls12>, ProvingKey<Fq, Scheme>) {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(4).build_global();
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, ChaCha20Rng::seed_from_u64(0x5eed));
    let (_, _, lookup_table) = values();
    let empty = MyCircuit {
        a: Value::unknown(),
        lookup_table,
    };
    let vk = keygen_vk_with_k(&params, &empty, K).expect("keygen_vk");
    let pk = keygen_pk(vk, &empty).expect("keygen_pk");
    (params, pk)
}

/// A proof of two instances of the circuit, with a fixed RNG seed.
pub fn prove(params: &ParamsKZG<Bls12>, pk: &ProvingKey<Fq, Scheme>) -> Vec<u8> {
    let (a, instance, lookup_table) = values();
    let circuit = MyCircuit {
        a: Value::known(a),
        lookup_table,
    };
    let rng = ChaCha20Rng::seed_from_u64(0x00c0ffee);
    let mut transcript = CircuitTranscript::<State>::init();
    create_proof::<Fq, Scheme, _, _>(
        params,
        pk,
        &[circuit.clone(), circuit],
        #[cfg(feature = "committed-instances")]
        0,
        &[&[&[instance]], &[&[instance]]],
        rng,
        &mut transcript,
    )
    .expect("create_proof");
    transcript.finalize()
}

/// Whether the proof verifies.
pub fn verifies(params: &ParamsKZG<Bls12>, pk: &ProvingKey<Fq, Scheme>, proof: &[u8]) -> bool {
    let (_, instance, _) = values();
    let pubinputs = [instance];
    let mut transcript = CircuitTranscript::<State>::init_from_bytes(proof);
    let Ok(guard) = prepare::<Fq, Scheme, _>(
        pk.get_vk(),
        #[cfg(feature = "committed-instances")]
        &[&[], &[]],
        &[&[&pubinputs[..]], &[&pubinputs[..]]],
        &mut transcript,
    ) else {
        return false;
    };
    guard.verify(&params.verifier_params()).is_ok()
}

/// BLAKE2b-256 of the proof, hex.
pub fn digest(proof: &[u8]) -> String {
    let h = blake2b_simd::Params::new().hash_length(32).hash(proof);
    h.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

/// Asserts two proofs are equal, naming the first differing byte.
pub fn assert_same_proof(a: &[u8], b: &[u8], what: &str) {
    if a != b {
        let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        panic!(
            "{what}: proofs differ (lengths {} and {}, first difference at byte {at})",
            a.len(),
            b.len()
        );
    }
}
