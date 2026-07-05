// Barnes-Hut gravity: approximating the force on each particle from everywhere else in
// the system in O(log n) node visits instead of O(n) pairwise checks.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';
import createGravityModule from './gravityWasm.mjs';

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

/**
 * Kicks off the (async) WASM module load - call once, early, e.g. when the worker starts.
 * computeGravity checks wasmReady itself and transparently falls back to the plain JS
 * implementation below until this resolves, so nothing has to block on it - the very first
 * frame or two of a session just run the JS path instead.
 */
export async function initGravityWasm() {
    if (wasmReady) return;
    wasmModule = await createGravityModule();
    wasmMaxParticles = wasmModule._get_max_particles();
    // Pointer getters are called once here rather than per-frame - see gravity.cpp's
    // comment on why these addresses (and the HEAPF32 view built from them) stay valid for
    // the module's entire lifetime (memory growth is disabled in build.sh specifically so
    // this caching is safe).
    wasmPosXOffset = wasmModule._get_pos_x_ptr() >> 2;
    wasmPosYOffset = wasmModule._get_pos_y_ptr() >> 2;
    wasmMassOffset = wasmModule._get_mass_ptr() >> 2;
    wasmRadiusOffset = wasmModule._get_radius_ptr() >> 2;
    wasmAccXOffset = wasmModule._get_acc_x_ptr() >> 2;
    wasmAccYOffset = wasmModule._get_acc_y_ptr() >> 2;
    wasmReady = true;
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
