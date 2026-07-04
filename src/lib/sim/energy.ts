// Kinetic/potential energy readouts for the debug panel.
import constants from '../constants.ts';

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
 */
function accumulatePotentialEnergy(system, i, node) {
    if (node.mass === 0) {
        return 0;
    }

    const dx = node.comX - system.posX[i];
    const dy = node.comY - system.posY[i];
    const distSq = dx * dx + dy * dy;

    if (!node.children) {
        if (node.occupant !== -1) {
            if (node.occupant === i) {
                return 0;
            }
            const j = node.occupant;
            const combinedRadius = system.radius[i] + system.radius[j];
            const softenedDistSq = distSq + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
            return -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * system.mass[j]) / Math.sqrt(softenedDistSq);
        }
        if (node.bucket) {
            let pe = 0;
            for (const j of node.bucket) {
                if (j === i) continue;
                const odx = system.posX[j] - system.posX[i];
                const ody = system.posY[j] - system.posY[i];
                const combinedRadius = system.radius[i] + system.radius[j];
                const softenedDistSq = odx * odx + ody * ody + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
                pe += -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * system.mass[j]) / Math.sqrt(softenedDistSq);
            }
            return pe;
        }
        return 0;
    }

    if (distSq > 0 && (node.size * node.size) / distSq < constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA) {
        const softenedDistSq = distSq + system.radius[i] * system.radius[i];
        return -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * node.mass) / Math.sqrt(softenedDistSq);
    }

    let pe = 0;
    for (const child of node.children) {
        pe += accumulatePotentialEnergy(system, i, child);
    }
    return pe;
}

/**
 * Total gravitational potential energy of the system, reusing the same tree already
 * built for gravity this frame rather than a third rebuild. Every particle traverses the
 * tree independently (same as applyTreeGravity), so each real pair's energy gets counted
 * once from each side - hence the final /2.
 */
export function computePotentialEnergy(system, tree) {
    let pe = 0;
    for (let i = 0; i < system.count; i++) {
        pe += accumulatePotentialEnergy(system, i, tree);
    }
    return pe / 2;
}
