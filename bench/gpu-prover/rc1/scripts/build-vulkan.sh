#!/usr/bin/env bash
set -euo pipefail
source bench/gpu-prover/rc1/scripts/common.sh

# The stock executable verifies the proofs and prepares the key cache (see build.sh);
# the Vulkan executable proves. glslang and SPIRV-Tools build the shaders: the build
# fails without them rather than producing an executable that finds no device.
mkdir -p "$bin_dir"
build=(cargo build --release --locked --offline -p midnight-gpu-prover-bench --bin rc1-bench)
"${build[@]}"
install -m755 "$CARGO_TARGET_DIR/release/rc1-bench" "$RC1_STOCK_BIN"
MIDNIGHT_VULKAN_REQUIRE_SPIRV=1 "${build[@]}" --features rc1-bench-vulkan
install -m755 "$CARGO_TARGET_DIR/release/rc1-bench" "$bin_dir/rc1-vulkan"
