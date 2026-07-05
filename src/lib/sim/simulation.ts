// Public entry point for the particle simulation: composes the individual pieces
// (particleSystem/quadtree/gravity/energy/merge/spawn) into the handful of calls
// Particles.vue actually needs to drive the sketch each frame.
import { kickAll, driftAll, resetAccelerationAll } from './particleSystem.ts';
import { mergeParticles } from './merge.ts';
import { collideParticles } from './collide.ts';
import { computeGravity } from './gravity.ts';
import { displayBody } from './particleRender.ts';

export { spawnParticles } from './spawn.ts';
export { computeCenterOfMass, recolorAll } from './particleSystem.ts';
export { computeKineticEnergy, computePotentialEnergy } from './energy.ts';
export { initGravityWasm, getGravityBackendLabel } from './gravity.ts';

/**
 * Runs one full physics frame on the given system, mutating it in place.
 *
 * With mergingEnabled false, colliding bodies bounce off each other (see collide.ts)
 * instead of combining into one - momentum- and energy-conserving elastic impulses keep
 * them from actually overlapping, so the swarm jostles into tight, dense clusters
 * wherever mass concentrates instead of consolidating into fewer, larger bodies over time.
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
        // Collision resolution builds its own tree for neighbor search (collide.ts, backed
        // by gravity.cpp's WASM tree or, as a fallback, spatialGrid.ts's JS grid) and
        // mutates positions/velocities resolving contacts - so by the time it returns, that
        // tree no longer matches the (now-changed) positions. Unlike the merge branch,
        // there's no valid tree to hand off here regardless of which backend collision
        // used; gravity always builds its own fresh one on the post-collision positions.
        collideParticles(system);
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
