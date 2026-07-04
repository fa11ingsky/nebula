// Public entry point for the particle simulation: composes the individual pieces
// (particleSystem/quadtree/gravity/energy/merge/spawn) into the handful of calls
// Particles.vue actually needs to drive the sketch each frame.
import { kickAll, driftAll, resetAccelerationAll } from './particleSystem.ts';
import { mergeParticles } from './merge.ts';
import { computeGravity } from './gravity.ts';
import { displayBody } from './particleRender.ts';

export { spawnParticles } from './spawn.ts';
export { computeCenterOfMass } from './particleSystem.ts';
export { computeKineticEnergy, computePotentialEnergy } from './energy.ts';

/**
 * Runs one full physics frame on the given system, mutating it in place.
 *
 * With mergingEnabled false, the merge-detection step is skipped entirely - colliding
 * bodies are never combined into one, so gravity alone (softened at close range, per
 * GRAVITY_SOFTENING_FACTOR) is what keeps them from actually overlapping. The visible
 * effect is a swarm that clumps into tight, dense bunches wherever mass concentrates,
 * instead of consolidating into fewer, larger bodies over time.
 * @returns {{state: object, gravityTree: object}}
 */
export function stepSimulation(system, explosions, mergingEnabled) {
    // Leapfrog ("kick-drift-kick") integration: half-kick with the acceleration already
    // sitting on each particle from the end of last frame, drift positions, recompute
    // gravity at the new positions, then apply the second half-kick. Same one gravity
    // evaluation per frame as plain Euler, far better energy behavior for orbital motion.
    kickAll(system, 0.5);
    driftAll(system);

    let gravityTree;
    if (mergingEnabled) {
        // Merges use their own quadtree built from pre-merge positions, and don't touch
        // acceleration at all - gravity is applied separately, right after. On frames where
        // nothing actually merged, the merge-phase tree is still exactly correct for gravity
        // too - positions didn't change - so it's reused instead of paying for a second full
        // rebuild.
        const mergeResult = mergeParticles(system, explosions);
        resetAccelerationAll(system);
        gravityTree = computeGravity(system, mergeResult.anyMerged ? null : mergeResult.tree);
    } else {
        resetAccelerationAll(system);
        gravityTree = computeGravity(system, null);
    }

    kickAll(system, 0.5);

    return { state: system, gravityTree };
}

export function displayAll(s, system, texturesEnabled) {
    for (let i = 0; i < system.count; i++) {
        displayBody(
            s,
            system.posX[i], system.posY[i],
            system.mass[i], system.radius[i],
            system.colorR[i], system.colorG[i], system.colorB[i],
            system.colorString[i],
            system.surface[i],
            texturesEnabled
        );
    }
}
