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
function applyTreeGravity(system, i, node) {
    if (node.mass === 0) {
        return;
    }

    const dx = node.comX - system.posX[i];
    const dy = node.comY - system.posY[i];
    const distSq = dx * dx + dy * dy;

    if (!node.children) {
        if (node.occupant !== -1) {
            if (node.occupant !== i) {
                applyDirectGravity(system, i, node.occupant, dx, dy, distSq);
            }
        } else if (node.bucket) {
            for (const j of node.bucket) {
                if (j === i) continue;
                const odx = system.posX[j] - system.posX[i];
                const ody = system.posY[j] - system.posY[i];
                applyDirectGravity(system, i, j, odx, ody, odx * odx + ody * ody);
            }
        }
        return;
    }

    // Opening-angle test: (size/distance)^2 < theta^2 -> approximate; otherwise descend.
    if (distSq > 0 && (node.size * node.size) / distSq < constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA) {
        applyAggregateGravity(system, i, node.mass, dx, dy, distSq);
        return;
    }

    for (const child of node.children) {
        applyTreeGravity(system, i, child);
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
    for (let i = 0; i < system.count; i++) {
        applyTreeGravity(system, i, gravityTree);
    }
    return gravityTree;
}
