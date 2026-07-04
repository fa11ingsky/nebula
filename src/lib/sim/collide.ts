// Collision response: the counterpart to merge.ts for when merging is disabled -
// particles that touch bounce off each other (conserving momentum, and losing some
// kinetic energy per COLLISION_RESTITUTION) instead of combining into one body.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';

/**
 * Collects every particle index within searchRadius of particle i into `out` - the same
 * quadtree range query as merge.ts's findMergeCandidates, kept as its own copy here since
 * merging and colliding are mutually exclusive per-frame behaviors (see simulation.ts)
 * with otherwise little in common.
 */
function findCollisionCandidates(system, node, i, searchRadius, out) {
    if (node.mass === 0) {
        return;
    }

    const px = system.posX[i];
    const py = system.posY[i];
    const closestX = Math.max(node.x, Math.min(px, node.x + node.size));
    const closestY = Math.max(node.y, Math.min(py, node.y + node.size));
    const dx = px - closestX;
    const dy = py - closestY;
    if (dx * dx + dy * dy > searchRadius * searchRadius) {
        return;
    }

    if (node.children) {
        for (const child of node.children) {
            findCollisionCandidates(system, child, i, searchRadius, out);
        }
        return;
    }

    if (node.occupant !== -1) {
        if (node.occupant !== i) {
            out.push(node.occupant);
        }
    } else if (node.bucket) {
        for (const j of node.bucket) {
            if (j !== i) out.push(j);
        }
    }
}

/**
 * Pushes two still-overlapping-but-already-separating bodies apart positionally, without
 * touching velocity - resolving a velocity impulse here would add energy instead of
 * removing it, since they're not actually approaching. Mass-weighted (the heavier body
 * moves less) so the pair's combined center of mass doesn't shift.
 */
function pushApart(system, i, j, weight1, weight2) {
    const dx = system.posX[j] - system.posX[i];
    const dy = system.posY[j] - system.posY[i];
    const touchDistance = system.radius[i] + system.radius[j];
    const distSq = dx * dx + dy * dy;

    if (distSq >= touchDistance * touchDistance || distSq < 1e-12) {
        return; // not actually overlapping right now, or exactly coincident - nothing to push apart
    }

    const dist = Math.sqrt(distSq);
    const correction = touchDistance / dist - 1; // scales (dx,dy) up to exactly touchDistance apart
    const pushX = dx * correction;
    const pushY = dy * correction;
    system.posX[i] -= weight1 * pushX;
    system.posY[i] -= weight1 * pushY;
    system.posX[j] += weight2 * pushX;
    system.posY[j] += weight2 * pushY;
}

/**
 * Applies a momentum-conserving impulse to i and j along the collision normal (nx, ny) -
 * the line connecting their centers at the moment of impact, not wherever they happen to
 * have ended up (see collideParticles). Only the velocity component along that normal is
 * affected, exactly like a real frictionless hard-sphere collision - any tangential
 * ("sliding past") component is left untouched. Equal and opposite by construction, so
 * momentum is conserved regardless of the mass ratio; kinetic energy is conserved only if
 * COLLISION_RESTITUTION is 1 (perfectly elastic) - at the default 0.5, half the closing
 * speed survives the bounce (reversed) and the rest is lost, same as a real inelastic
 * collision losing energy to heat/deformation.
 *
 * Also advances position for whatever fraction of this frame's drift remained after the
 * moment of impact, using the new velocity - otherwise the bounce would only visibly take
 * effect a full frame late, since position was already fully drifted at the old velocity
 * before this function runs.
 */
function resolveImpulse(system, i, j, nx, ny, remainingT) {
    const relVxIJ = system.velX[i] - system.velX[j];
    const relVyIJ = system.velY[i] - system.velY[j];
    const vn = relVxIJ * nx + relVyIJ * ny; // positive = i approaching j along the normal

    const m1 = system.mass[i];
    const m2 = system.mass[j];
    const impulse = ((1 + constants.COLLISION_RESTITUTION) * vn) / (1 / m1 + 1 / m2);

    const oldVelXi = system.velX[i];
    const oldVelYi = system.velY[i];
    const oldVelXj = system.velX[j];
    const oldVelYj = system.velY[j];

    const newVelXi = oldVelXi - (impulse / m1) * nx;
    const newVelYi = oldVelYi - (impulse / m1) * ny;
    const newVelXj = oldVelXj + (impulse / m2) * nx;
    const newVelYj = oldVelYj + (impulse / m2) * ny;

    system.velX[i] = newVelXi;
    system.velY[i] = newVelYi;
    system.velX[j] = newVelXj;
    system.velY[j] = newVelYj;

    // Current position already drifted the full frame at the old velocity; correcting it
    // to reflect the new velocity for just the remaining fraction of the step is the same
    // as adding remainingT * (new - old) - the part of the drift that should have used the
    // post-bounce velocity instead.
    system.posX[i] += remainingT * (newVelXi - oldVelXi);
    system.posY[i] += remainingT * (newVelYi - oldVelYi);
    system.posX[j] += remainingT * (newVelXj - oldVelXj);
    system.posY[j] += remainingT * (newVelYj - oldVelYj);
}

/**
 * Finds particles whose paths crossed within touching distance (the sum of their radii -
 * their surfaces actually meeting, unlike merge.ts's smaller merge threshold) this frame
 * and resolves the contact, either as a bounce (still approaching) or a pure positional
 * separation (already moving apart but still overlapping).
 *
 * Uses continuous (swept) collision detection for the same reason merge.ts does: checking
 * only where i and j ended up after this frame's drift misses a fast, close flyby that
 * crosses within touching distance mid-frame but ends the frame well past it in either
 * direction. This can't just test the *closest approach* distance the way merge.ts does,
 * though - resolving an impulse also needs a normal direction and a sense of whether the
 * pair is still approaching, and at the exact point of closest approach the relative
 * velocity is, by definition, always perpendicular to the separation vector (the calculus
 * condition for a minimum), which would make every swept contact look like a
 * zero-closing-speed graze. Instead this solves for the *time of impact* - the instant
 * separation first reaches the touch distance, while the two are still actually
 * approaching - via the standard sphere-sweep quadratic: |P0 + tV|^2 = touchDistance^2,
 * solved for the smallest valid root in [0,1].
 */
export function collideParticles(system) {
    const tree = buildQuadtree(system);
    const count = system.count;

    // Upper bound on how far apart two particles could possibly be and still touch,
    // so the range query never misses a legitimate candidate - see merge.ts's identical
    // use of globalMaxRadius for the full derivation.
    let globalMaxRadius = 0;
    for (let i = 0; i < count; i++) {
        if (system.radius[i] > globalMaxRadius) globalMaxRadius = system.radius[i];
    }

    const candidates = [];
    // Marks pairs already resolved this frame (packed as min*capacity+max, a unique
    // integer per unordered pair) so a pair found from both sides - once as i seeking j,
    // once as j seeking i, since colliding (unlike merging) never removes either particle
    // - only gets resolved once. A single particle can still resolve against several
    // different neighbors in the same frame, matching how a body in a dense, jostling
    // pile can be touching more than one neighbor at once.
    const resolvedPairs = new Set();

    for (let i = 0; i < count; i++) {
        const speedI = Math.sqrt(system.velX[i] * system.velX[i] + system.velY[i] * system.velY[i]);
        candidates.length = 0;
        findCollisionCandidates(system, tree, i, system.radius[i] + globalMaxRadius + speedI, candidates);

        for (const j of candidates) {
            const pairKey = i < j ? i * system.capacity + j : j * system.capacity + i;
            if (resolvedPairs.has(pairKey)) {
                continue;
            }

            const relVx = system.velX[j] - system.velX[i]; // V, "j relative to i"
            const relVy = system.velY[j] - system.velY[i];
            const relSpeedSq = relVx * relVx + relVy * relVy; // a

            const endDx = system.posX[j] - system.posX[i];
            const endDy = system.posY[j] - system.posY[i];
            // Relative position before this frame's drift: the end position minus the
            // relative displacement drift just applied - reconstructs "before" without
            // needing to have stored it separately, since drift moves each particle by
            // exactly its own velocity.
            const startDx = endDx - relVx; // P0
            const startDy = endDy - relVy;

            const touchDistance = system.radius[i] + system.radius[j];
            const c = startDx * startDx + startDy * startDy - touchDistance * touchDistance;

            let t; // fraction of this frame's drift at which contact occurred
            let contactDx;
            let contactDy;
            if (c <= 0) {
                // Already inside touch distance at the start of this frame's drift (most
                // often a pair still settling from a previous frame without having fully
                // separated again) - resolve using that starting separation directly
                // rather than searching for a "first contact" that already happened.
                t = 0;
                contactDx = startDx;
                contactDy = startDy;
            } else if (relSpeedSq < 1e-9) {
                continue; // not already touching, and no relative motion to close the gap
            } else {
                const b = 2 * (startDx * relVx + startDy * relVy);
                const discriminant = b * b - 4 * relSpeedSq * c;
                if (discriminant < 0) {
                    continue; // paths never come within touch distance at all this frame
                }
                // Smaller root: the instant separation first reaches touchDistance, while
                // still approaching - not the closest-approach point (which would always
                // show zero closing speed, since relative velocity is perpendicular to
                // separation there by construction).
                t = (-b - Math.sqrt(discriminant)) / (2 * relSpeedSq);
                if (t < 0 || t > 1) {
                    continue; // the crossing happens outside this frame's time window
                }
                contactDx = startDx + t * relVx;
                contactDy = startDy + t * relVy;
            }

            const m1 = system.mass[i];
            const m2 = system.mass[j];
            const weight1 = m2 / (m1 + m2); // how much of a positional correction body i absorbs
            const weight2 = m1 / (m1 + m2); // ...and body j - heavier body moves less

            // closingSpeed<0 means approaching (contactDx/Dy and relVx/Vy are both in the
            // "j relative to i" convention, so this is d.v in that convention - negative
            // exactly when the separation is shrinking).
            const closingSpeed = contactDx * relVx + contactDy * relVy;

            if (closingSpeed >= 0) {
                // Already moving apart (or exactly tangential) at the moment contact began -
                // if drift still carried them through each other despite that, just push
                // them apart positionally; a velocity impulse here would add energy rather
                // than remove it.
                pushApart(system, i, j, weight1, weight2);
                resolvedPairs.add(pairKey);
                continue;
            }

            const dist = Math.sqrt(contactDx * contactDx + contactDy * contactDy) || 1e-6; // guard against exactly-coincident particles
            const nx = contactDx / dist;
            const ny = contactDy / dist;

            resolveImpulse(system, i, j, nx, ny, 1 - t);
            resolvedPairs.add(pairKey);
        }
    }
}
