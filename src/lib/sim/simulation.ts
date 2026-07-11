// Public entry point for the particle simulation: composes the individual pieces
// (particleSystem/quadtree/gravity/energy/merge/spawn) into the handful of calls
// Particles.vue actually needs to drive the sketch each frame.
import { kickAll, driftAll, resetAccelerationAll } from './particleSystem.ts';
import { mergeParticles } from './merge.ts';
import { collideParticles } from './collide.ts';
import { computeGravity } from './gravity.ts';
import { computeGravityPM } from './pmGravity.ts';
import { displayBody } from './particleRender.ts';
import { densityRamp } from './colors.ts';

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
 *
 * `pm` (optional) switches the gravity solve from the Barnes-Hut tree to the Particle-Mesh
 * solver (pmGravity.ts - the 'pm' entry in constants.GRAVITY_SOLVER_OPTIONS): pass
 * { grid, table, scratch, G, pairSofteningFactor } built by the worker at spawn time.
 * Everything else (leapfrog structure, merging, collision) is identical; only the force
 * model changes. gravityTree is null in that case - PM has no tree for the energy readout
 * to reuse (energy.ts builds its own fallback on demand).
 * @returns {{state: object, gravityTree: object}}
 */
export function stepSimulation(system, explosions, mergingEnabled, pm = null) {
    // Leapfrog ("kick-drift-kick") integration: half-kick with the acceleration already
    // sitting on each particle from the end of last frame, drift positions, recompute
    // gravity at the new positions, then apply the second half-kick. Same one gravity
    // evaluation per frame as plain Euler, far better energy behavior for orbital motion.
    kickAll(system, 0.5);
    driftAll(system);

    let gravityTree = null;
    if (mergingEnabled) {
        // Merges use their own quadtree built from pre-merge positions, and don't touch
        // acceleration at all - gravity is applied separately, right after. On frames where
        // nothing actually merged, the merge-phase tree is still exactly correct for gravity
        // too - positions didn't change - so it's reused instead of paying for a second full
        // rebuild.
        const mergeResult = mergeParticles(system, explosions);
        resetAccelerationAll(system);
        if (pm) {
            // The PM mesh solve zeroes and rebuilds acceleration itself (see
            // computeGravityPMMesh) - the constant external GRAVITY field
            // resetAccelerationAll just seeded gets overwritten, which only matters if
            // GRAVITY.X/Y is ever set nonzero (it defaults to zero).
            computeGravityPM(system, pm.grid, pm.table, pm.G, pm.pairSofteningFactor, pm.scratch);
        } else {
            gravityTree = computeGravity(system, mergeResult.anyMerged ? null : mergeResult.tree);
        }
    } else {
        // Collision resolution builds its own tree for neighbor search (collide.ts, backed
        // by gravity.cpp's WASM tree or, as a fallback, spatialGrid.ts's JS grid) and
        // mutates positions/velocities resolving contacts - so by the time it returns, that
        // tree no longer matches the (now-changed) positions. Unlike the merge branch,
        // there's no valid tree to hand off here regardless of which backend collision
        // used; gravity always builds its own fresh one on the post-collision positions.
        collideParticles(system);
        resetAccelerationAll(system);
        if (pm) {
            computeGravityPM(system, pm.grid, pm.table, pm.G, pm.pairSofteningFactor, pm.scratch);
        } else {
            gravityTree = computeGravity(system, null);
        }
    }

    kickAll(system, 0.5);

    return { state: system, gravityTree };
}

/**
 * `densityColors` colors every body by colors.ts's densityRamp over its current local
 * crowding (system.density[i], populated by collide.ts each frame) instead of its own
 * mass-gradient/fixed color - see particleRender.ts's displayBody for why this overrides
 * texturesEnabled rather than combining with it.
 */
export function displayAll(s, system, texturesEnabled, densityColors = false) {
    for (let i = 0; i < system.count; i++) {
        displayBody(
            s,
            system.posX[i], system.posY[i],
            system.mass[i], system.radius[i],
            system.colorR[i], system.colorG[i], system.colorB[i],
            system.colorString[i],
            system.surface[i],
            texturesEnabled,
            densityColors ? densityRamp(system.density[i]) : null
        );
    }
}
