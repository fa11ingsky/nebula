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
 * Builds the swarm, optionally with an extra dense cluster of particles packed into the
 * center (see CENTRAL_MASS_FRACTION/CENTRAL_CLUSTER_RADIUS_FRACTION - not one dominant
 * body, just ordinary particles spawned within a much smaller radius), then sets up the
 * system's initial rotation. Shared by both the first load and Restart so the two never
 * drift out of sync with each other.
 * @returns {{state: object, worldCenter: {x: number, y: number}}}
 */
export function spawnParticles(s, includeCentralMass) {
    const centralParticleCount = includeCentralMass
        ? Math.round(constants.TOTAL_PARTICLES * constants.CENTRAL_MASS_FRACTION)
        : 0;
    const capacity = constants.TOTAL_PARTICLES + centralParticleCount;
    const system = createParticleSystem(capacity);

    const centerX = s.width / 2;
    const centerY = s.height / 2;
    const spawnRadius = Math.min(s.width, s.height) / 2 * constants.SPAWN_RADIUS_FRACTION;
    const particleMass = constants.MAX_MASS / constants.TOTAL_PARTICLES;

    for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
        const spawn = randomSpawnPoint(s, centerX, centerY, spawnRadius);
        addParticle(system, spawn.x, spawn.y, particleMass);
    }

    if (includeCentralMass) {
        const clusterRadius = spawnRadius * constants.CENTRAL_CLUSTER_RADIUS_FRACTION;
        const clusterStart = system.count;
        for (let i = 0; i < centralParticleCount; i++) {
            const spawn = randomSpawnPoint(s, centerX, centerY, clusterRadius);
            addParticle(system, spawn.x, spawn.y, particleMass);
        }

        // randomSpawnPoint scatters each cluster particle independently, so their own
        // average position lands near (centerX, centerY) only statistically, not exactly -
        // for a small cluster particle count especially, an unlucky draw can leave the
        // whole cluster visibly off-center. Since every cluster particle has equal mass,
        // shifting all of them by the same delta re-centers their centroid exactly on the
        // arena's center without disturbing the cluster's shape or density.
        let sumX = 0;
        let sumY = 0;
        for (let i = clusterStart; i < system.count; i++) {
            sumX += system.posX[i];
            sumY += system.posY[i];
        }
        const shiftX = centerX - sumX / centralParticleCount;
        const shiftY = centerY - sumY / centralParticleCount;
        for (let i = clusterStart; i < system.count; i++) {
            system.posX[i] += shiftX;
            system.posY[i] += shiftY;
        }
    }

    initializeAngularMomentum(system, constants.TOTAL_ANGULAR_MOMENTUM);

    return { state: system, worldCenter: { x: centerX, y: centerY } };
}
