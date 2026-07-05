// Kinetic/potential energy readouts for the debug panel.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';

/**
 * Total kinetic energy of the system: sum(0.5 * m * v^2). Exact, O(n) - no approximation
 * needed here, unlike potential energy below.
 */
export function computeKineticEnergy(system) {
    let ke = 0;
    for (let i = 0; i < system.count; i++) {
        const vx = system.velX[i];
        const vy = system.velY[i];
        ke += 0.5 * system.mass[i] * (vx * vx + vy * vy);
    }
    return ke;
}

/**
 * Accumulates the gravitational potential energy between particle i and everything else
 * in the tree, using the exact same direct/aggregate split (and opening-angle criterion)
 * as gravity.ts's applyTreeGravity - so this is a Barnes-Hut approximation of PE, not an
 * exact sum, consistent with the fact that the forces driving this simulation are
 * approximated the same way. Softened the same way as force so a pair well inside the
 * merge threshold doesn't produce a singular energy value.
 *
 * Iterative rather than recursive - see gravity.ts's applyTreeGravity for the full
 * rationale (next-pointer traversal, precomputed thetaSq, multiplication instead of
 * division in the opening-angle test).
 */
function accumulatePotentialEnergy(system, i, tree, thetaSq) {
    const px = system.posX[i];
    const py = system.posY[i];
    let node = 0;
    let pe = 0;

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
                    const combinedRadius = system.radius[i] + system.radius[occ];
                    const softenedDistSq = distSq + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
                    pe += -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * system.mass[occ]) / Math.sqrt(softenedDistSq);
                }
            } else if (tree.bucket[node]) {
                for (const j of tree.bucket[node]) {
                    if (j === i) continue;
                    const odx = system.posX[j] - px;
                    const ody = system.posY[j] - py;
                    const combinedRadius = system.radius[i] + system.radius[j];
                    const softenedDistSq = odx * odx + ody * ody + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
                    pe += -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * system.mass[j]) / Math.sqrt(softenedDistSq);
                }
            }
            node = tree.next[node];
        } else if (tree.size[node] * tree.size[node] < distSq * thetaSq) {
            const softenedDistSq = distSq + system.radius[i] * system.radius[i];
            pe += -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * mass) / Math.sqrt(softenedDistSq);
            node = tree.next[node];
        } else {
            node = tree.children[node];
        }
    }

    return pe;
}

/**
 * Total gravitational potential energy of the system, reusing the same tree already
 * built for gravity this frame rather than a third rebuild. Every particle traverses the
 * tree independently (same as applyTreeGravity), so each real pair's energy gets counted
 * once from each side - hence the final /2.
 *
 * tree can be null when gravity.ts's computeGravity ran on the WASM path instead of the JS
 * one - there's no JS-shaped tree to hand back in that case (see computeGravity's own
 * comment), so this builds its own on demand instead. Only affects this debug-panel-only
 * readout, not the hot path: it only runs while the debug panel is actually visible.
 */
export function computePotentialEnergy(system, tree) {
    const peTree = tree || buildQuadtree(system);
    let pe = 0;
    const thetaSq = constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA;
    for (let i = 0; i < system.count; i++) {
        pe += accumulatePotentialEnergy(system, i, peTree, thetaSq);
    }
    return pe / 2;
}
