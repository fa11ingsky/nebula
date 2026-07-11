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
    const offsetX = new Float32Array(system.count);
    const offsetY = new Float32Array(system.count);
    for (let i = 0; i < system.count; i++) {
        if (system.fixed[i]) continue; // never assigned a velocity - see the loop below
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

    // A fixed particle (see particleSystem.ts) is deliberately left at velocity 0 rather
    // than the v=omega*r' every other particle gets - not just because driftAll would
    // ignore it anyway, but because assigning it a nonzero velocity would misreport
    // kinetic energy in the debug panel for a body that can never actually move. The
    // "total linear momentum exactly zero" proof in this function's own doc comment
    // assumes every particle follows v=omega*r'; excluding a fixed body breaks that
    // exactly, in proportion to how far it sits from the center of mass - expected and
    // accepted here, the same way a real collision with an immovable wall doesn't
    // conserve just the ball's own momentum.
    for (let i = 0; i < system.count; i++) {
        if (system.fixed[i]) continue;
        system.velX[i] = -omega * offsetY[i];
        system.velY[i] = omega * offsetX[i];
    }
}

/**
 * Builds the swarm, optionally with one additional central body - a single fixed particle
 * at the exact arena center, holding CENTRAL_MASS_FRACTION of MAX_MASS, that gravity and
 * collisions can act on (it attracts and can be bounced off of) but that never itself
 * moves - see particleSystem.ts's `fixed` field. Then sets up the system's initial
 * rotation. Shared by both the first load and Restart so the two never drift out of sync
 * with each other.
 *
 * mergingEnabled controls initial particle color (see colors.ts's getDisplayColorForMass) -
 * defaults to true so existing callers that don't pass it (tests, mainly) keep the
 * original mass-gradient coloring rather than silently switching to white.
 * @returns {{state: object, worldCenter: {x: number, y: number}}}
 */
export function spawnParticles(s, includeCentralMass, mergingEnabled = true, lite = false) {
    const capacity = constants.TOTAL_PARTICLES + (includeCentralMass ? 1 : 0);
    const system = createParticleSystem(capacity);

    const centerX = s.width / 2;
    const centerY = s.height / 2;
    const spawnRadius = Math.min(s.width, s.height) / 2 * constants.SPAWN_RADIUS_FRACTION;
    const particleMass = constants.MAX_MASS / constants.TOTAL_PARTICLES;
    const particleRadius = Math.sqrt(particleMass);

    for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
        const spawn = randomSpawnPoint(s, centerX, centerY, spawnRadius);
        addParticle(system, spawn.x, spawn.y, particleMass, mergingEnabled, false, null, lite);
    }

    if (includeCentralMass) {
        const centralMass = constants.MAX_MASS * constants.CENTRAL_MASS_FRACTION;
        // Radius deliberately pinned to an ordinary swarm particle's size (not
        // sqrt(centralMass), what it would get by default) - it needs a large mass for
        // gravity to treat it as the dominant body, but should still read as an
        // ordinary-sized particle rather than a giant sphere, and (with merging off, so
        // nothing ever changes mass or radius again after spawn - see merge.ts, the only
        // place that reassigns radius) this stays true for the entire session.
        addParticle(system, centerX, centerY, centralMass, mergingEnabled, true, particleRadius, lite);
    }

    initializeAngularMomentum(system, constants.TOTAL_ANGULAR_MOMENTUM);

    return { state: system, worldCenter: { x: centerX, y: centerY } };
}
