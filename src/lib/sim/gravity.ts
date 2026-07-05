// Barnes-Hut gravity: approximating the force on each particle from everywhere else in
// the system in O(log n) node visits instead of O(n) pairwise checks.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';

// The hot path (build a fresh tree, then force-traverse it for every particle) ported to
// WebAssembly - see src/wasm/gravity.cpp. Only that specific case moves to WASM: when the
// caller already has a JS-built tree to reuse (merge-mode's "nothing merged this frame, so
// last tree is still valid" optimization - see simulation.ts), this stays on the plain JS
// path below, since there's no cheap way to hand a JS quadtree's data to WASM without
// rebuilding it anyway - and rebuilding is exactly the expensive case this exists to avoid.
/** @type {any} */
let wasmModule = null;
let wasmReady = false;
let wasmMaxParticles = 0;
let wasmPosXOffset = 0, wasmPosYOffset = 0, wasmMassOffset = 0, wasmRadiusOffset = 0;
let wasmAccXOffset = 0, wasmAccYOffset = 0;
let wasmCandidateBufferOffset = 0;
let wasmThreaded = false;

/**
 * Kicks off the (async) WASM module load - call once, early, e.g. when the worker starts.
 * computeGravity checks wasmReady itself and transparently falls back to the plain JS
 * implementation below until this resolves, so nothing has to block on it - the very first
 * frame or two of a session just run the JS path instead.
 *
 * Prefers the multi-threaded build (`npm run build:wasm:threaded` in package.json -
 * compute_gravity's force traversal split across a std::thread per core) when the page is
 * cross-origin isolated, since that's the one precondition for SharedArrayBuffer (and so
 * WASM threads) to exist at all - see public/_headers (Netlify) and vite.config.ts (local
 * dev/preview) for where that isolation actually gets turned on. Falls back to the plain
 * single-threaded build (`npm run build:wasm`) both when isolation isn't available and if
 * the threaded build fails to instantiate for some other reason (e.g. a browser without
 * WASM threads support) - either way this should never be the reason the app fails to run,
 * only the reason it runs on one core instead of several.
 */
export async function initGravityWasm() {
    if (wasmReady) return;

    if (typeof self !== 'undefined' && self.crossOriginIsolated) {
        try {
            const { default: createThreadedGravityModule } = await import('./gravityWasm.threaded.mjs');
            wasmModule = await createThreadedGravityModule();
            wasmThreaded = true;
        } catch (err) {
            console.warn('Threaded WASM gravity module failed to load, falling back to single-threaded:', err);
        }
    }

    if (!wasmModule) {
        const { default: createGravityModule } = await import('./gravityWasm.mjs');
        wasmModule = await createGravityModule();
        wasmThreaded = false;
    }

    wasmMaxParticles = wasmModule._get_max_particles();
    // Pointer getters are called once here rather than per-frame - see gravity.cpp's
    // comment on why these addresses (and the HEAPF32 view built from them) stay valid for
    // the module's entire lifetime (memory growth is disabled in both builds specifically
    // so this caching is safe).
    wasmPosXOffset = wasmModule._get_pos_x_ptr() >> 2;
    wasmPosYOffset = wasmModule._get_pos_y_ptr() >> 2;
    wasmMassOffset = wasmModule._get_mass_ptr() >> 2;
    wasmRadiusOffset = wasmModule._get_radius_ptr() >> 2;
    wasmAccXOffset = wasmModule._get_acc_x_ptr() >> 2;
    wasmAccYOffset = wasmModule._get_acc_y_ptr() >> 2;
    wasmCandidateBufferOffset = wasmModule._get_candidate_buffer_ptr() >> 2;
    wasmReady = true;
}

/**
 * Whether the WASM spatial module (gravity.cpp - it backs both gravity and collide.ts's
 * broad-phase search, see that file's header comment) has finished loading. collide.ts
 * checks this itself rather than going through computeGravity, since collision runs before
 * gravity every frame and needs its own readiness/fallback decision.
 */
export function isSpatialWasmReady() {
    return wasmReady;
}

/** Particle-count ceiling for the WASM module - same fixed capacity for both gravity and collision, since they share one MAX_PARTICLES in gravity.cpp. */
export function getSpatialWasmMaxParticles() {
    return wasmMaxParticles;
}

/**
 * Builds a fresh tree in WASM for collide.ts's broad-phase search - see gravity.cpp's
 * build_collision_tree for why this takes its own leaf capacity (COLLISION_TREE_LEAF_CAPACITY)
 * and max depth (COLLISION_TREE_MAX_DEPTH) separate from gravity's own
 * QUADTREE_LEAF_CAPACITY/QUADTREE_MAX_DEPTH - measured directly, sharing gravity's (shallow)
 * max depth left dense clusters (the distributed central-mass feature especially) stuck at
 * 70+ particles per leaf, since there wasn't enough depth budget left to actually subdivide
 * down to COLLISION_TREE_LEAF_CAPACITY. It's safe to reuse the same input buffers
 * compute_gravity uses (collision and gravity never run concurrently within a frame). Only
 * copies position/mass in, not radius/acceleration - the collision tree never reads either.
 */
export function buildCollisionTreeWasm(system) {
    const count = system.count;
    const heap = wasmModule.HEAPF32;
    heap.set(system.posX.subarray(0, count), wasmPosXOffset);
    heap.set(system.posY.subarray(0, count), wasmPosYOffset);
    heap.set(system.mass.subarray(0, count), wasmMassOffset);
    wasmModule._build_collision_tree(count, constants.COLLISION_TREE_MAX_DEPTH, constants.COLLISION_TREE_LEAF_CAPACITY);
}

/**
 * Collects every particle within searchRadius of particle i into `out` - the WASM-backed
 * counterpart to quadtree.ts's findNearbyParticles/spatialGrid.ts's findNearbyInGrid, same
 * broad-phase contract (candidates are a superset of the true answer; collide.ts's own
 * swept-collision math does the exact filtering). Must be called after
 * buildCollisionTreeWasm this frame. Capped at get_max_candidates() results per call (see
 * gravity.cpp's MAX_CANDIDATES) - degrades gracefully rather than growing unbounded, since
 * an uncapped result is exactly the runaway-cost failure mode a tree was built to avoid.
 */
export function findNearbyWasm(system, i, searchRadius, out) {
    const count = wasmModule._find_nearby_collision_candidates(i, searchRadius);
    const candidates = wasmModule.HEAP32.subarray(wasmCandidateBufferOffset, wasmCandidateBufferOffset + count);
    for (let k = 0; k < count; k++) {
        out.push(candidates[k]);
    }
}

/**
 * Debug-panel-only readout of which gravity path is actually active right now: the plain
 * JS Barnes-Hut fallback (either the WASM module hasn't finished loading yet, or this
 * particular frame reused a JS-built tree - see computeGravity), the single-threaded WASM
 * build, or the multi-threaded one (only possible on a cross-origin-isolated page - see
 * initGravityWasm).
 */
export function getGravityBackendLabel() {
    if (!wasmReady) return 'js (wasm loading)';
    return wasmThreaded ? 'wasm (threaded)' : 'wasm (single-thread)';
}

/**
 * Copies position/mass/radius into the WASM module's own memory, runs its build-tree-then-
 * traverse-it kernel, and adds the resulting pure-gravity acceleration onto the system's
 * existing accX/accY (which already carries the constant external GRAVITY field from
 * resetAccelerationAll - the WASM side has no notion of that field, it only ever computes
 * the gravitational contribution, hence += rather than = on the way back out). The four
 * input copies and two output copies are plain typed-array memcpy-equivalents - at 50k
 * particles that's a few hundred KB, microseconds next to the hundreds of milliseconds of
 * compute this replaces.
 */
function computeGravityWasm(system) {
    const count = system.count;
    const heap = wasmModule.HEAPF32;
    heap.set(system.posX.subarray(0, count), wasmPosXOffset);
    heap.set(system.posY.subarray(0, count), wasmPosYOffset);
    heap.set(system.mass.subarray(0, count), wasmMassOffset);
    heap.set(system.radius.subarray(0, count), wasmRadiusOffset);

    wasmModule._compute_gravity(count, constants.GRAVITATIONAL_CONSTANT, constants.GRAVITY_SOFTENING_FACTOR, constants.BARNES_HUT_THETA, constants.QUADTREE_MAX_DEPTH, constants.QUADTREE_LEAF_CAPACITY);

    const outAccX = heap.subarray(wasmAccXOffset, wasmAccXOffset + count);
    const outAccY = heap.subarray(wasmAccYOffset, wasmAccYOffset + count);
    for (let i = 0; i < count; i++) {
        system.accX[i] += outAccX[i];
        system.accY[i] += outAccY[i];
    }
}

/**
 * Accumulates gravitational acceleration onto particle i from one specific other body j,
 * softened over their combined radius (see GRAVITY_SOFTENING_FACTOR) so the force stays
 * finite even at zero separation. Only updates i's own acceleration - see applyTreeGravity
 * for why that's still momentum-conserving overall.
 */
function applyDirectGravity(system, i, j, dx, dy, distSq) {
    const combinedRadius = system.radius[i] + system.radius[j];
    const softenedDistSq = distSq + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
    const dist = Math.sqrt(softenedDistSq);
    const scalar = (constants.GRAVITATIONAL_CONSTANT * system.mass[j]) / (softenedDistSq * dist);
    system.accX[i] += dx * scalar;
    system.accY[i] += dy * scalar;
}

/**
 * Same as applyDirectGravity, but for treating an entire distant subtree as a single
 * point mass at its center of mass. There's no specific "other body" to size a softening
 * length from here, but the opening-angle test in applyTreeGravity already guarantees
 * this is only used when the node is comfortably far away, so a minimal softening off
 * just this particle's own radius is enough of a safety net.
 */
function applyAggregateGravity(system, i, otherMass, dx, dy, distSq) {
    const r = system.radius[i];
    const softenedDistSq = distSq + r * r;
    const dist = Math.sqrt(softenedDistSq);
    const scalar = (constants.GRAVITATIONAL_CONSTANT * otherMass) / (softenedDistSq * dist);
    system.accX[i] += dx * scalar;
    system.accY[i] += dy * scalar;
}

/**
 * Accumulates the total gravitational acceleration on particle i from everywhere else in
 * the tree (Barnes-Hut). For a node that's far enough away relative to its size (the
 * opening-angle test), the whole subtree is treated as one point mass instead of
 * descending into it - O(log n) node visits per particle instead of O(n).
 *
 * Iterative rather than recursive: `tree`'s next-pointers (see quadtree.ts) let this walk
 * the whole traversal as a single flat loop - skip a subtree (it's a leaf, or approved for
 * aggregation) by jumping to `next`; otherwise descend into `children`. thetaSq is passed
 * in already squared and precomputed once per computeGravity() call rather than
 * recomputed on every node visit, and the opening-angle test is written as a
 * multiplication (`size^2 < distSq*thetaSq`) instead of a division - cheaper, and (unlike
 * the division form) doesn't need a separate distSq>0 guard, since size^2 is always
 * positive and distSq=0 correctly fails the test either way, forcing a descend.
 *
 * On momentum conservation: every particle runs this traversal independently, and only
 * ever updates its OWN acceleration (never anyone else's). For two individual particles
 * resolved down to actual leaves, both sides independently compute the same physics from
 * their own perspective, which is exactly equal-and-opposite by construction - so
 * near-field interactions stay (numerically, near-exactly) momentum-conserving, the same
 * as a direct pairwise sum. Far-field interactions approximated as a cluster are the
 * exception: only the querying particle gets a force from "the cluster," with no single
 * reaction applied back to the many individual bodies inside it. That's the real
 * accuracy/speed trade this algorithm makes - total momentum is very close to conserved,
 * not exact to the bit, in exchange for going from O(n^2) to O(n log n).
 */
function applyTreeGravity(system, i, tree, thetaSq) {
    const px = system.posX[i];
    const py = system.posY[i];
    let node = 0;

    while (node !== -1) {
        const mass = tree.mass[node];
        if (mass === 0) {
            node = tree.next[node];
            continue;
        }

        const dx = tree.comX[node] - px;
        const dy = tree.comY[node] - py;
        const distSq = dx * dx + dy * dy;

        if (tree.children[node] === -1) {
            const occ = tree.occupant[node];
            if (occ !== -1) {
                if (occ !== i) {
                    applyDirectGravity(system, i, occ, dx, dy, distSq);
                }
            } else if (tree.bucket[node]) {
                for (const j of tree.bucket[node]) {
                    if (j === i) continue;
                    const odx = system.posX[j] - px;
                    const ody = system.posY[j] - py;
                    applyDirectGravity(system, i, j, odx, ody, odx * odx + ody * ody);
                }
            }
            node = tree.next[node];
        } else if (tree.size[node] * tree.size[node] < distSq * thetaSq) {
            applyAggregateGravity(system, i, mass, dx, dy, distSq);
            node = tree.next[node];
        } else {
            node = tree.children[node];
        }
    }
}

/**
 * Applies gravity to every particle, building a tree first unless one is already
 * provided. Call after resetAccelerationAll() and before the second half-kick.
 *
 * Accepting a pre-built tree lets the caller skip a second full rebuild on frames where
 * nothing merged: positions haven't changed since the merge-detection tree was built, so
 * that tree is already exactly correct for gravity too. Rebuilding it anyway would be
 * pure waste - and became a bigger one once merges got rarer (see merge.ts's notes on
 * MIN_MERGE_VELOCITY), since more frames now have nothing merge on them at all.
 *
 * When there's no tree to reuse, this is the "build fresh and traverse for everyone" case
 * that dominates frame time at high particle counts - routed to the WASM kernel above once
 * it's loaded and the particle count fits its fixed capacity, falling back to the plain JS
 * Barnes-Hut implementation otherwise (WASM still loading, or an unusually large run that
 * exceeds gravity.cpp's MAX_PARTICLES). Returns null in the WASM case rather than a JS tree
 * object, since there's nothing JS-shaped to hand back - computePotentialEnergy in
 * energy.ts builds its own fallback tree on demand for the (debug-panel-only, not hot path)
 * cases that need one.
 */
export function computeGravity(system, tree) {
    if (tree) {
        const thetaSq = constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA;
        for (let i = 0; i < system.count; i++) {
            applyTreeGravity(system, i, tree, thetaSq);
        }
        return tree;
    }

    if (wasmReady && system.count <= wasmMaxParticles) {
        computeGravityWasm(system);
        return null;
    }

    const gravityTree = buildQuadtree(system);
    const thetaSq = constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA;
    for (let i = 0; i < system.count; i++) {
        applyTreeGravity(system, i, gravityTree, thetaSq);
    }
    return gravityTree;
}
