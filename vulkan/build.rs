// This file is part of MIDNIGHT-ZK.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Build of the Vulkan compute shaders.
//!
//! Every kernel of [`KERNELS`] is compiled twice, once per arithmetic variant
//! (`MONT_INT64=1`: variant A, 64-bit accumulators; `MONT_INT64=0`: variant
//! B, 32-bit words only), in three steps:
//!
//! 1. GLSL to unoptimised SPIR-V 1.6 (Vulkan 1.3) with `glslangValidator` or
//!    `glslc`;
//! 2. the functions the kernel lists in `keep` get the SPIR-V function
//!    control `DontInline`;
//! 3. `spirv-opt -O`, which inlines every other function and fully unrolls
//!    the `[[unroll]]` loops.
//!
//! The SPIR-V is embedded with `include_bytes!` through
//! `$OUT_DIR/shaders.rs`, and the cfg `midnight_vulkan_spirv` is set. A
//! variant B module declaring the `Int64` capability is a build error.
//! Kernels marked `test` are compiled only with the feature `test-support`.
//!
//! Without the tools the crate builds with no SPIR-V (a cargo warning, the cfg
//! unset), and the device context declines every device.
//!
//! Environment:
//! - `MIDNIGHT_VULKAN_GLSL`: the GLSL compiler (`glslangValidator` or
//!   `glslc`, told apart by file name); default: the first of the two on
//!   `PATH`.
//! - `MIDNIGHT_VULKAN_SPIRV_OPT`: `spirv-opt`; default: `spirv-opt` on
//!   `PATH`.
//! - `MIDNIGHT_VULKAN_REQUIRE_SPIRV=1`: missing tools are a build error.
use std::{
    env,
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

/// A kernel: one entry point, compiled from `source` with `defines`.
struct Kernel {
    name: &'static str,
    source: &'static str,
    defines: &'static [&'static str],
    /// Functions kept out of line (`DontInline`), by GLSL name.
    keep: &'static [&'static str],
    /// Compiled only with the feature `test-support`.
    test: bool,
}

const KERNELS: &[Kernel] = &[
    Kernel {
        name: "diag_iota",
        source: "diag_iota.comp",
        defines: &[],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "field_test_fr",
        source: "field_test.comp",
        defines: &["FIELD_FR"],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "field_test_fp",
        source: "field_test.comp",
        defines: &["FIELD_FP"],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "msm_test_g1",
        source: "msm_test_g1.comp",
        defines: &[],
        keep: &["g1_mul"],
        test: true,
    },
    Kernel {
        name: "msm_test_mask",
        source: "msm_test_mask.comp",
        defines: &[],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "bench_fr_mul",
        source: "bench.comp",
        defines: &["BENCH_FR_MUL"],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "bench_fp_mul",
        source: "bench.comp",
        defines: &["BENCH_FP_MUL"],
        keep: &[],
        test: true,
    },
    Kernel {
        name: "bench_g1_madd",
        source: "bench.comp",
        defines: &["BENCH_G1_MADD"],
        keep: &["g1_mul"],
        test: true,
    },
    Kernel {
        name: "bench_g1_madd_inline",
        source: "bench.comp",
        defines: &["BENCH_G1_MADD"],
        keep: &[],
        test: true,
    },
];

/// The arithmetic variants: (file suffix, `MONT_INT64`).
const VARIANTS: [(&str, u32); 2] = [("int64", 1), ("int32", 0)];

const INSTALL: &str = "install glslang (`glslangValidator`) or shaderc (`glslc`), and \
                       SPIRV-Tools (`spirv-opt`): the Vulkan SDK, Homebrew `glslang`, or the \
                       Linux packages `glslang-tools` and `spirv-tools`";

enum Frontend {
    Glslang(PathBuf),
    Glslc(PathBuf),
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    let include = shaders.join("include");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shaders.display());
    println!("cargo:rerun-if-changed={}", include.display());
    for dir in [&shaders, &include] {
        for e in std::fs::read_dir(dir).expect("vulkan/shaders exists").flatten() {
            if e.path().is_file() {
                println!("cargo:rerun-if-changed={}", e.path().display());
            }
        }
    }
    for v in [
        "MIDNIGHT_VULKAN_GLSL",
        "MIDNIGHT_VULKAN_SPIRV_OPT",
        "MIDNIGHT_VULKAN_REQUIRE_SPIRV",
        "PATH",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rustc-check-cfg=cfg(midnight_vulkan_spirv)");

    let test_support = env::var_os("CARGO_FEATURE_TEST_SUPPORT").is_some();
    let kernels: Vec<&Kernel> = KERNELS.iter().filter(|k| test_support || !k.test).collect();
    let mut rs = String::from("pub(crate) static SHADERS: &[Shader] = &[\n");
    match tools() {
        Ok((frontend, opt)) => {
            std::thread::scope(|s| {
                let jobs: Vec<_> = kernels
                    .iter()
                    .flat_map(|k| VARIANTS.iter().map(move |v| (*k, *v)))
                    .map(|(k, v)| {
                        let (frontend, opt, out, shaders, include) =
                            (&frontend, &opt, &out, &shaders, &include);
                        s.spawn(move || compile(k, v, frontend, opt, shaders, include, out))
                    })
                    .collect();
                let errors: Vec<String> =
                    jobs.into_iter().filter_map(|j| j.join().unwrap().err()).collect();
                if !errors.is_empty() {
                    panic!(
                        "midnight-vulkan: shader build failed:\n{}",
                        errors.join("\n")
                    );
                }
            });
            for k in &kernels {
                let path = |suffix: &str| out.join(format!("{}.{suffix}.spv", k.name));
                writeln!(
                    rs,
                    "    Shader {{ name: {:?}, int64: include_bytes!({:?}), int32: include_bytes!({:?}) }},",
                    k.name,
                    path("int64"),
                    path("int32"),
                )
                .unwrap();
            }
            println!("cargo:rustc-cfg=midnight_vulkan_spirv");
        }
        Err(why) => {
            if env::var("MIDNIGHT_VULKAN_REQUIRE_SPIRV").as_deref() == Ok("1") {
                panic!("midnight-vulkan: {why} (MIDNIGHT_VULKAN_REQUIRE_SPIRV=1)");
            }
            println!(
                "cargo:warning=midnight-vulkan: {why}; building without SPIR-V, the Vulkan backend \
                 declines every device"
            );
        }
    }
    rs.push_str("];\n");
    std::fs::write(out.join("shaders.rs"), rs).unwrap();
}

/// The GLSL compiler (`MIDNIGHT_VULKAN_GLSL`, or `glslangValidator` or
/// `glslc` on `PATH`) and `spirv-opt`, or why they are missing.
fn tools() -> Result<(Frontend, PathBuf), String> {
    let path = tool("MIDNIGHT_VULKAN_GLSL", &["glslangValidator", "glslc"]).ok_or_else(|| {
        format!(
            "no GLSL compiler was found (glslangValidator or glslc); {INSTALL}, or set \
             MIDNIGHT_VULKAN_GLSL"
        )
    })?;
    let opt = tool("MIDNIGHT_VULKAN_SPIRV_OPT", &["spirv-opt"]).ok_or_else(|| {
        format!("spirv-opt was not found; {INSTALL}, or set MIDNIGHT_VULKAN_SPIRV_OPT")
    })?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let frontend = if stem.starts_with("glslc") {
        Frontend::Glslc(path)
    } else {
        Frontend::Glslang(path)
    };
    Ok((frontend, opt))
}

/// `$var` if set, else the first of `names` found on `PATH`.
fn tool(var: &str, names: &[&str]) -> Option<PathBuf> {
    if let Some(p) = env::var_os(var) {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let dirs: Vec<PathBuf> =
        env::var_os("PATH").map(|p| env::split_paths(&p).collect()).unwrap_or_default();
    names.iter().find_map(|n| {
        dirs.iter()
            .map(|d| d.join(n))
            .find(|p| p.is_file())
            .or_else(|| dirs.iter().map(|d| d.join(format!("{n}.exe"))).find(|p| p.is_file()))
    })
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    let o = cmd
        .output()
        .map_err(|e| format!("{what}: could not run {:?}: {e}", cmd.get_program()))?;
    if !o.status.success() {
        return Err(format!(
            "{what}: {}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ));
    }
    Ok(())
}

fn compile(
    k: &Kernel,
    (suffix, int64): (&str, u32),
    frontend: &Frontend,
    opt: &Path,
    shaders: &Path,
    include: &Path,
    out: &Path,
) -> Result<(), String> {
    let what = format!("{} ({suffix})", k.name);
    let src = shaders.join(k.source);
    let raw = out.join(format!("{}.{suffix}.raw.spv", k.name));
    let dst = out.join(format!("{}.{suffix}.spv", k.name));
    let mut defines = vec![format!("-DMONT_INT64={int64}")];
    defines.extend(k.defines.iter().map(|d| format!("-D{d}")));
    let mut cmd = match frontend {
        Frontend::Glslang(p) => {
            let mut c = Command::new(p);
            c.args(["-V", "--target-env", "vulkan1.3", "--quiet", "-S", "comp"]);
            c
        }
        Frontend::Glslc(p) => {
            let mut c = Command::new(p);
            c.args(["--target-env=vulkan1.3", "-O0", "-fshader-stage=compute"]);
            c
        }
    };
    cmd.arg(format!("-I{}", include.display()))
        .args(&defines)
        .arg("-o")
        .arg(&raw)
        .arg(&src);
    run(&mut cmd, &what)?;

    let mut words = read_words(&raw).map_err(|e| format!("{what}: {e}"))?;
    keep_out_of_line(&mut words, k.keep).map_err(|e| format!("{what}: {e}"))?;
    write_words(&raw, &words).map_err(|e| format!("{what}: {e}"))?;
    run(
        Command::new(opt)
            .args(["-O", "--target-env=vulkan1.3"])
            .arg(&raw)
            .arg("-o")
            .arg(&dst),
        &what,
    )?;

    let words = read_words(&dst).map_err(|e| format!("{what}: {e}"))?;
    if int64 == 0 && declares_int64(&words) {
        return Err(format!("{what}: variant B declares the Int64 capability"));
    }
    Ok(())
}

fn read_words(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 || bytes.len() < 20 || bytes[..4] != 0x0723_0203u32.to_le_bytes() {
        return Err(format!("{} is not little-endian SPIR-V", path.display()));
    }
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

fn write_words(path: &Path, words: &[u32]) -> Result<(), String> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The instructions of a module after its header: (opcode, operands).
fn instructions(words: &[u32]) -> impl Iterator<Item = (usize, u16, &[u32])> {
    let mut at = 5;
    std::iter::from_fn(move || {
        let w = *words.get(at)?;
        let len = (w >> 16) as usize;
        if len == 0 || at + len > words.len() {
            return None;
        }
        let r = (at, (w & 0xffff) as u16, &words[at + 1..at + len]);
        at += len;
        Some(r)
    })
}

const OP_NAME: u16 = 5;
const OP_CAPABILITY: u16 = 17;
const OP_FUNCTION: u16 = 54;
const CAPABILITY_INT64: u32 = 11;
const FUNCTION_CONTROL_INLINE: u32 = 0x1;
const FUNCTION_CONTROL_DONT_INLINE: u32 = 0x2;

/// Sets `DontInline` on every function whose GLSL name is in `keep`
/// (glslang and glslc name a function `name(<mangled parameters>`).
fn keep_out_of_line(words: &mut [u32], keep: &[&str]) -> Result<(), String> {
    if keep.is_empty() {
        return Ok(());
    }
    let mut ids = Vec::new();
    let mut found = vec![false; keep.len()];
    for (_, op, args) in instructions(words) {
        if op != OP_NAME || args.is_empty() {
            continue;
        }
        let bytes: Vec<u8> = args[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
        let name = bytes.split(|b| *b == 0).next().unwrap_or_default();
        let base = name.split(|b| *b == b'(').next().unwrap_or_default();
        if let Some(i) = keep.iter().position(|k| k.as_bytes() == base) {
            ids.push(args[0]);
            found[i] = true;
        }
    }
    if let Some(i) = found.iter().position(|f| !f) {
        return Err(format!("no function {} to keep out of line", keep[i]));
    }
    let at: Vec<usize> = instructions(words)
        .filter(|(_, op, args)| *op == OP_FUNCTION && args.len() >= 3 && ids.contains(&args[1]))
        .map(|(at, ..)| at)
        .collect();
    for at in at {
        // OpFunction <result type> <result id> <control> <function type>
        let control = &mut words[at + 3];
        *control = (*control & !FUNCTION_CONTROL_INLINE) | FUNCTION_CONTROL_DONT_INLINE;
    }
    Ok(())
}

fn declares_int64(words: &[u32]) -> bool {
    instructions(words)
        .any(|(_, op, args)| op == OP_CAPABILITY && args.first() == Some(&CAPABILITY_INT64))
}
