// Runs before every `vite build` (see package.json's prebuild script) to catch a specific
// deploy-breaking mistake early: the WASM gravity/collision module (src/wasm/gravity.cpp)
// is compiled locally via `npm run build:wasm`/`build:wasm:threaded` and its output is
// committed to the repo like any other generated asset - Netlify (and GitHub Pages before
// it) never runs Emscripten, they just bundle whatever's already sitting in
// src/lib/sim/gravityWasm*. That means it's possible to edit gravity.cpp, forget to
// recompile, or recompile but forget to `git add` the result, and have `vite build` succeed
// anyway - it has no way to know the artifacts are stale or missing, since it just treats
// them as regular source files. This check exists so that failure mode is a loud, immediate
// build error instead of a silently degraded (or broken) deployed app.
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const simDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'lib', 'sim');

// Only the plain build is strictly required - gravity.ts falls back to it whenever the
// threaded one is missing, unready, or fails to instantiate (e.g. no cross-origin
// isolation). The threaded files are checked too, but only warned about: their absence
// degrades performance (single-threaded gravity/collision instead of multi-threaded), not
// correctness, so it shouldn't fail the whole build the way a missing required file does.
const required = ['gravityWasm.mjs'];
const optional = ['gravityWasm.threaded.mjs', 'gravityWasm.threaded.wasm'];

const missingRequired = required.filter((f) => !existsSync(join(simDir, f)));
const missingOptional = optional.filter((f) => !existsSync(join(simDir, f)));

if (missingRequired.length > 0) {
    console.error(
        `\nMissing required WASM artifact(s): ${missingRequired.join(', ')}\n` +
        `Run "npm run build:wasm" (requires Emscripten - see src/wasm/gravity.cpp) and commit the output.\n`
    );
    process.exit(1);
}

if (missingOptional.length > 0) {
    console.warn(
        `\nWarning: missing optional WASM artifact(s): ${missingOptional.join(', ')}\n` +
        `The app will still build and run, but gravity/collision will fall back to the ` +
        `single-threaded WASM build even on a cross-origin-isolated page. Run ` +
        `"npm run build:wasm:threaded" and commit the output to fix.\n`
    );
}
