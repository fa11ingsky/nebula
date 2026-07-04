// Building the initial swarm: placing particles, and setting up initial velocities so
// the system starts with a controlled net angular momentum and exactly zero net momentum.
import constants from '../constants.ts';
import { createParticleSystem, addParticle, computeCenterOfMass } from './particleSystem.ts';

/**
 * Picks a uniformly random point within radius of (centerX, centerY), purely to shape
 * the initial cluster - there's no boundary/wall tied to this.
 */
export function randomSpawnPoint(s, centerX, centerY, radius) {
    const angle = s.random(Math.PI * 2);
    const r = radius * Math.sqrt(s.random(1));
    return { x: centerX + r * Math.cos(angle), y: centerY + r * Math.sin(angle) };
}

/**
 * Sets every particle's initial velocity to a rigid-body-style rotation about the
 * system's own center of mass (velocity proportional to distance from the center of
 * mass, all sharing one angular velocity omega), which guarantees two things at once:
 *  - total linear momentum is exactly zero: sum(m_i * v_i) = omega x sum(m_i * r_i'),
 *    and mass-weighted offsets from the center of mass always sum to zero by definition.
 *  - total angular momentum about that center of mass - and, since linear momentum is
 *    zero, about any other fixed point too - equals exactly targetL, via the standard
 *    rigid-body relation L = I * omega.
 */
function initializeAngularMomentum(system, targetL) {
    const com = computeCenterOfMass(system);

    let momentOfInertia = 0;
    const offsetX = new Float64Array(system.count);
    const offsetY = new Float64Array(system.count);
    for (let i = 0; i < system.count; i++) {
        const dx = system.posX[i] - com.x;
        const dy = system.posY[i] - com.y;
        offsetX[i] = dx;
        offsetY[i] = dy;
        momentOfInertia += system.mass[i] * (dx * dx + dy * dy);
    }

    if (momentOfInertia < 1e-6) {
        return; // every particle sits at the center of mass; no rotation is meaningful
    }

    const omega = targetL / momentOfInertia;

    for (let i = 0; i < system.count; i++) {
        system.velX[i] = -omega * offsetY[i];
        system.velY[i] = omega * offsetX[i];
    }
}

/**
 * Builds the swarm, optionally followed by a heavy central body, then sets up the
 * system's initial rotation. Shared by both the first load and Restart so the two never
 * drift out of sync with each other.
 * @returns {{state: object, worldCenter: {x: number, y: number}}}
 */
export function spawnParticles(s, includeCentralMass) {
    const capacity = constants.TOTAL_PARTICLES + (includeCentralMass ? 1 : 0);
    const system = createParticleSystem(capacity);

    const centerX = s.width / 2;
    const centerY = s.height / 2;
    const spawnRadius = Math.min(s.width, s.height) / 2 * constants.SPAWN_RADIUS_FRACTION;

    for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
        const spawn = randomSpawnPoint(s, centerX, centerY, spawnRadius);
        addParticle(system, spawn.x, spawn.y, constants.MAX_MASS / constants.TOTAL_PARTICLES);
    }

    if (includeCentralMass) {
        addParticle(system, centerX, centerY, constants.MAX_MASS * constants.CENTRAL_MASS_FRACTION);
    }

    initializeAngularMomentum(system, constants.TOTAL_ANGULAR_MOMENTUM);

    return { state: system, worldCenter: { x: centerX, y: centerY } };
}
