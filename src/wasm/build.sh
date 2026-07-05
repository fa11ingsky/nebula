#!/usr/bin/env bash
# Compiles gravity.cpp to a self-contained ES module (WASM binary embedded as base64 via
# SINGLE_FILE) that simulation code can just `import` like any other module - no separate
# .wasm asset for Vite's bundler to resolve, which matters here since the consumer
# (simulation.worker.ts) runs inside a Web Worker, not the main document.
#
# Re-run this manually after editing gravity.cpp - the output is checked into the repo
# (src/lib/sim/gravityWasm.mjs) like any other generated artifact, so building the app
# doesn't require every contributor (or the GitHub Pages deploy step) to have Emscripten
# installed - only whoever is actively changing the C++ needs the toolchain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

emcc gravity.cpp -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createGravityModule \
    -s ENVIRONMENT=web,worker,node \
    -s ALLOW_MEMORY_GROWTH=0 \
    -s INITIAL_MEMORY=67108864 \
    -s SINGLE_FILE=1 \
    -s FILESYSTEM=0 \
    -s EXPORTED_FUNCTIONS=_compute_gravity,_get_pos_x_ptr,_get_pos_y_ptr,_get_mass_ptr,_get_radius_ptr,_get_acc_x_ptr,_get_acc_y_ptr,_get_max_particles \
    -s EXPORTED_RUNTIME_METHODS=HEAPF32 \
    -o ../lib/sim/gravityWasm.mjs

echo "Built src/lib/sim/gravityWasm.mjs"
