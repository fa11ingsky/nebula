// Barnes-Hut gravity: approximating the force on each particle from everywhere else in
// the system in O(log n) node visits instead of O(n) pairwise checks.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';

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
 */
export function computeGravity(system, tree) {
    const gravityTree = tree || buildQuadtree(system);
    const thetaSq = constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA;
    for (let i = 0; i < system.count; i++) {
        applyTreeGravity(system, i, gravityTree, thetaSq);
    }
    return gravityTree;
}
