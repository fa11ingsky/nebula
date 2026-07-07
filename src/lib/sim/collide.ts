// Collision response: the counterpart to merge.ts for when merging is disabled -
// particles that touch bounce off each other (conserving momentum, and losing some
// kinetic energy per COLLISION_RESTITUTION) instead of combining into one body.
import constants from '../constants.ts';
import { buildSpatialGrid, findNearbyInGrid } from './spatialGrid.ts';
import { isSpatialWasmReady, getSpatialWasmMaxParticles, buildCollisionTreeWasm, findAllCollisionCandidatesWasm, getCollisionCandidatesWasm } from './gravity.ts';

// Snapshot of every particle's velocity at the moment this frame's drift actually ran,
// reused across frames and grown as needed (same pattern as the quadtree's node store) -
// see collideParticles for why this has to be frozen rather than read live.
let origVelX = new Float32Array(0);
let origVelY = new Float32Array(0);

function ensureVelocitySnapshotCapacity(count) {
    if (count <= origVelX.length) {
        return;
    }
    origVelX = new Float32Array(count);
    origVelY = new Float32Array(count);
}

/**
 * Softened gravitational potential energy between i and j at the given separation -
 * matches energy.ts's direct-pair PE formula exactly, so the "energy budget" this
 * function reasons about is the same quantity the debug panel actually reports.
 */
function softenedPairPE(system, i, j, dist) {
    const combinedRadius = system.radius[i] + system.radius[j];
    const softenedDistSq = dist * dist + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
    return -(constants.GRAVITATIONAL_CONSTANT * system.mass[i] * system.mass[j]) / Math.sqrt(softenedDistSq);
}

/**
 * Enforces the hard non-overlap constraint (two particles' surfaces may never
 * interpenetrate) for a pair found overlapping that isn't already being separated by an
 * approaching-pair bounce this frame. Separating two gravitationally-bound bodies always
 * increases their mutual potential energy - moving them apart by fiat, with no
 * compensating change anywhere else, creates that energy from nothing (this is exactly
 * the bug an earlier, simpler version of this function had: proven, via an isolated
 * test, to inject energy on every call). Instead, this pays for the PE increase by
 * removing exactly that much kinetic energy from the pair's own relative motion along
 * the separation normal, via the same impulse mechanics as resolveImpulse - so the
 * position change is never free.
 *
 * If the pair doesn't have enough of *that specific* energy (relative velocity along the
 * normal) to afford separating all the way out to touchDistance, this separates them as
 * far as the available budget actually pays for and no further - a pair truly at rest
 * relative to each other while overlapping can't be pulled apart without inventing
 * energy, so it's left slightly overlapping for this frame (gravity's very next kick
 * will pull them into a real, energy-accounted approach/bounce next frame instead).
 *
 * If either body is fixed (particleSystem.ts - currently just the optional central mass),
 * it's treated as infinitely massive for this split: see resolveImpulse's comment on
 * invM1/invM2 for the derivation. All of the separation and all of the energy payment
 * comes from the other body; the fixed one doesn't move and its velocity is untouched.
 */
function resolveOverlap(system, i, j) {
    const dx = system.posX[j] - system.posX[i];
    const dy = system.posY[j] - system.posY[i];
    const dist = Math.sqrt(dx * dx + dy * dy) || 1e-6;
    const touchDistance = system.radius[i] + system.radius[j] + constants.COLLISION_SURFACE_GAP;

    if (dist >= touchDistance) {
        return; // not actually overlapping - nothing to do
    }

    const nx = dx / dist;
    const ny = dy / dist;

    const m1 = system.mass[i];
    const m2 = system.mass[j];
    const invM1 = system.fixed[i] ? 0 : 1 / m1;
    const invM2 = system.fixed[j] ? 0 : 1 / m2;
    const mu = 1 / (invM1 + invM2); // reduced mass (equals the other body's own mass if one side is fixed)

    const relVx = system.velX[i] - system.velX[j];
    const relVy = system.velY[i] - system.velY[j];
    const vn = relVx * nx + relVy * ny; // relative speed along the normal (i toward j is positive)

    const availableEnergy = 0.5 * mu * vn * vn;
    const fullSeparationCost = softenedPairPE(system, i, j, touchDistance) - softenedPairPE(system, i, j, dist);

    let targetDist;
    if (availableEnergy >= fullSeparationCost) {
        targetDist = touchDistance;
    } else {
        // Solve for the largest r1 the available budget actually affords: softenedPairPE
        // is monotonically increasing in distance, so this has exactly one solution.
        const combinedRadius = system.radius[i] + system.radius[j];
        const softeningSq = combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
        const gm1m2 = constants.GRAVITATIONAL_CONSTANT * m1 * m2;
        const invSqrtR0 = 1 / Math.sqrt(dist * dist + softeningSq);
        const k = invSqrtR0 - availableEnergy / gm1m2;
        if (k <= 0) {
            targetDist = touchDistance; // budget covers even more than the full gap - cap there anyway
        } else {
            targetDist = Math.sqrt(Math.max(1 / (k * k) - softeningSq, 0));
        }
    }

    if (targetDist <= dist) {
        return; // no relative motion along the normal to spend - can't afford any separation right now
    }

    // Move the pair apart along the normal, mass-weighted so the heavier body moves less
    // and their combined center of mass doesn't shift - weight1 = invM1/(invM1+invM2) is
    // algebraically identical to m2/(m1+m2) when neither body is fixed (multiply both
    // terms by m1*m2 to see it), but also correctly goes to exactly 0 when body i is fixed
    // (invM1=0), putting the entire shift on the other body instead of the usual split.
    const weight1 = invM1 / (invM1 + invM2);
    const weight2 = invM2 / (invM1 + invM2);
    const shift = targetDist - dist;
    system.posX[i] -= weight1 * shift * nx;
    system.posY[i] -= weight1 * shift * ny;
    system.posX[j] += weight2 * shift * nx;
    system.posY[j] += weight2 * shift * ny;

    // Pay for exactly the potential energy just spent by removing that much kinetic
    // energy from the pair's relative motion along the normal - same quadratic as
    // resolveImpulse's restitution formula, just solved for "cancel this specific energy
    // cost" instead of "reverse this fraction of the closing speed". Keeps the same sign
    // of relative motion (just slower), rather than flipping it, since a positional
    // correction "braking" a separation is the physically sensible direction.
    const spentEnergy = softenedPairPE(system, i, j, targetDist) - softenedPairPE(system, i, j, dist);
    const discriminant = Math.max(vn * vn - (2 * spentEnergy) / mu, 0);
    const vnNew = Math.sign(vn) * Math.sqrt(discriminant);
    const impulse = mu * (vn - vnNew);

    system.velX[i] -= impulse * invM1 * nx;
    system.velY[i] -= impulse * invM1 * ny;
    system.velX[j] += impulse * invM2 * nx;
    system.velY[j] += impulse * invM2 * ny;
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
 *
 * If either body is fixed (particleSystem.ts - currently just the optional central mass),
 * it's treated as infinitely massive: 1/m becomes 0 for that body (invM1/invM2 below), the
 * classic "collision with an immovable wall" limit. Taking the general impulse formula
 * impulse = (1+e)*vn / (1/m1 + 1/m2) to that limit for body i gives
 * impulse = (1+e)*vn*m2 exactly, so the fixed body's own velocity term (impulse/m1, i.e.
 * impulse*invM1) is exactly 0 - not approximately, since invM1 is exactly 0, not just a
 * very large m1 - while the other body gets the full, standard "bounce off a wall" result.
 */
function resolveImpulse(system, i, j, nx, ny, remainingT) {
    const relVxIJ = system.velX[i] - system.velX[j];
    const relVyIJ = system.velY[i] - system.velY[j];
    const vn = relVxIJ * nx + relVyIJ * ny; // positive = i approaching j along the normal

    const m1 = system.mass[i];
    const m2 = system.mass[j];
    const invM1 = system.fixed[i] ? 0 : 1 / m1;
    const invM2 = system.fixed[j] ? 0 : 1 / m2;
    const impulse = ((1 + constants.COLLISION_RESTITUTION) * vn) / (invM1 + invM2);

    const oldVelXi = system.velX[i];
    const oldVelYi = system.velY[i];
    const oldVelXj = system.velX[j];
    const oldVelYj = system.velY[j];

    const newVelXi = oldVelXi - impulse * invM1 * nx;
    const newVelYi = oldVelYi - impulse * invM1 * ny;
    const newVelXj = oldVelXj + impulse * invM2 * nx;
    const newVelYj = oldVelYj + impulse * invM2 * ny;

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
 *
 * The swept reconstruction (relative velocity, and the pre-drift position derived from
 * it) always reads a snapshot of every velocity taken at entry, never system.velX/velY
 * live - a particle already bounced earlier in this same pass has a *different* current
 * velocity than the one that actually produced this frame's drift, and reconstructing
 * "where was it before this frame" from the wrong velocity gives the wrong answer,
 * silently missing collisions that really happened. resolveImpulse itself still reads
 * live velocities, since the bounce itself should correctly account for an earlier
 * bounce this same frame (standard sequential impulse resolution) - only the geometry
 * reconstruction needs the frozen snapshot.
 *
 * Candidate lookup prefers the WASM tree (gravity.cpp's build_collision_tree/
 * find_all_collision_candidates - see that file's header comment) over the JS uniform grid
 * (spatialGrid.ts, kept as the fallback for when WASM isn't ready yet or the particle count
 * exceeds its fixed capacity). The grid was originally built to replace an earlier
 * quadtree-based version of this search - a bounded-radius query a uniform grid answers in
 * a handful of direct cell lookups, versus a tree's O(log n) descent with pruning tests
 * paying for hierarchy the query doesn't need - but a fixed cell size doesn't adapt to
 * density: profiling with the distributed central-mass feature enabled showed 50+
 * particles landing in a single grid cell sized for ~4, degrading that "handful of lookups"
 * into an effectively O(n) scan for every particle in the dense region. A tree doesn't have
 * that failure mode (it just subdivides the dense region further), and WASM keeps it fast
 * even so.
 *
 * The WASM path runs the search for every particle in one batched, multi-threaded call
 * (findAllCollisionCandidatesWasm) rather than once per particle - profiling showed this
 * search, not the swept-collision math/resolution below, is the majority of a collision
 * frame's cost, and it has the same read-only-tree, write-only-to-own-output shape that
 * makes gravity's own traversal safe to split across threads. The JS grid fallback stays
 * per-particle (findNearbyInGrid) - there's no thread pool to hand that off to on the JS
 * side regardless.
 */
export function collideParticles(system) {
    const count = system.count;
    const useWasm = isSpatialWasmReady() && count <= getSpatialWasmMaxParticles();
    let grid = null;
    if (useWasm) {
        buildCollisionTreeWasm(system);
    } else {
        grid = buildSpatialGrid(system);
    }

    ensureVelocitySnapshotCapacity(count);
    for (let i = 0; i < count; i++) {
        origVelX[i] = system.velX[i];
        origVelY[i] = system.velY[i];
    }

    // Upper bound on how far apart two particles could possibly be and still touch,
    // so the range query never misses a legitimate candidate - see merge.ts's identical
    // use of globalMaxRadius for the full derivation.
    let globalMaxRadius = 0;
    for (let i = 0; i < count; i++) {
        if (system.radius[i] > globalMaxRadius) globalMaxRadius = system.radius[i];
    }

    if (useWasm) {
        findAllCollisionCandidatesWasm(system, origVelX, origVelY, globalMaxRadius, constants.COLLISION_SURFACE_GAP);
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
        candidates.length = 0;
        if (useWasm) {
            // Search radius (including this particle's own speed padding) was already
            // computed inside find_all_collision_candidates - this is just reading back
            // the result of that batched call, not making a fresh WASM call per particle.
            getCollisionCandidatesWasm(i, candidates);
        } else {
            const speedI = Math.sqrt(origVelX[i] * origVelX[i] + origVelY[i] * origVelY[i]);
            const searchRadius = system.radius[i] + globalMaxRadius + constants.COLLISION_SURFACE_GAP + speedI;
            findNearbyInGrid(grid, system, i, searchRadius, candidates);
        }

        for (const j of candidates) {
            const pairKey = i < j ? i * system.capacity + j : j * system.capacity + i;
            if (resolvedPairs.has(pairKey)) {
                continue;
            }

            const relVx = origVelX[j] - origVelX[i]; // V, "j relative to i"
            const relVy = origVelY[j] - origVelY[i];
            const relSpeedSq = relVx * relVx + relVy * relVy; // a

            const endDx = system.posX[j] - system.posX[i];
            const endDy = system.posY[j] - system.posY[i];
            // Relative position before this frame's drift: the end position minus the
            // relative displacement drift just applied - reconstructs "before" without
            // needing to have stored it separately, since drift moves each particle by
            // exactly its own velocity.
            const startDx = endDx - relVx; // P0
            const startDy = endDy - relVy;

            const touchDistance = system.radius[i] + system.radius[j] + constants.COLLISION_SURFACE_GAP;
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

            // closingSpeed<0 means approaching (contactDx/Dy and relVx/Vy are both in the
            // "j relative to i" convention, so this is d.v in that convention - negative
            // exactly when the separation is shrinking).
            const closingSpeed = contactDx * relVx + contactDy * relVy;

            if (closingSpeed >= 0) {
                // Already moving apart (or exactly tangential) at the moment contact began -
                // no bounce needed (their own velocity is already carrying them apart), but
                // if they're still currently overlapping regardless, the non-overlap
                // constraint still has to be enforced - see resolveOverlap for why that
                // can't just be a positional shove (that was an earlier, buggy version of
                // this function, proven to inject energy from nothing every time it ran).
                resolveOverlap(system, i, j);
                resolvedPairs.add(pairKey);
                continue;
            }

            const dist = Math.sqrt(contactDx * contactDx + contactDy * contactDy) || 1e-6; // guard against exactly-coincident particles
            const nx = contactDx / dist;
            const ny = contactDy / dist;

            resolveImpulse(system, i, j, nx, ny, 1 - t);
            // Belt-and-suspenders: resolveImpulse's post-bounce position advancement
            // should already put them at or past touchDistance, but only covers the
            // remaining fraction of this frame's drift - if that fraction was too small
            // to fully clear the overlap, enforce the constraint directly rather than
            // letting a sliver of penetration carry into the next frame.
            resolveOverlap(system, i, j);
            resolvedPairs.add(pairKey);
        }
    }
}
