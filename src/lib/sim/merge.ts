// Collision detection/resolution: finding particles close enough (and moving fast
// enough) to fuse, combining them, and compacting the system afterward.
import constants from '../constants.ts';
import { buildQuadtree } from './quadtree.ts';
import { copyParticle } from './particleSystem.ts';
import { getColorForMass } from './colors.ts';
import { generateSurfaceFeatures } from './surfaceFeatures.ts';
import { Explosion } from './explosion.ts';

/**
 * Collects every particle index within searchRadius of particle i into `out`, pruning
 * subtrees whose bounding box can't possibly contain a point that close (closest-point-
 * on-box test).
 */
function findMergeCandidates(system, node, i, searchRadius, out) {
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
            findMergeCandidates(system, child, i, searchRadius, out);
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
 * Combines particle j into particle i in place (i survives and grows; j is left stale
 * and expected to be dropped by the caller's compaction pass).
 */
function mergeInto(system, i, j) {
    const m1 = system.mass[i];
    const m2 = system.mass[j];
    const totalMass = m1 + m2;

    // Momentum-weighted velocity.
    system.velX[i] = (system.velX[i] * m1 + system.velX[j] * m2) / totalMass;
    system.velY[i] = (system.velY[i] * m1 + system.velY[j] * m2) / totalMass;

    // Weighted by mass so the merged body lands at the true center of mass instead of
    // the geometric midpoint - otherwise merging unequal masses would inject spurious
    // angular momentum into the system.
    system.posX[i] = (system.posX[i] * m1 + system.posX[j] * m2) / totalMass;
    system.posY[i] = (system.posY[i] * m1 + system.posY[j] * m2) / totalMass;

    system.radius[i] = Math.sqrt(totalMass);
    system.mass[i] = totalMass;

    const color = getColorForMass(totalMass);
    system.colorR[i] = color[0];
    system.colorG[i] = color[1];
    system.colorB[i] = color[2];
    system.colorString[i] = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    system.surface[i] = generateSurfaceFeatures(totalMass);
}

/**
 * Checks for merges via a quadtree range query instead of an all-pairs scan, compacting
 * the system in place so the live set stays contiguous at [0, count). Every merge appends
 * an Explosion to `explosions`, sized off the merged body's combined mass. Pure merge
 * detection only - gravity is applied separately (see gravity.ts's computeGravity).
 * @param {object} system - the ParticleSystem, mutated in place
 * @param {Array} explosions - collision flashes, mutated in place
 * @returns {{tree: QuadTreeNode, anyMerged: boolean}} - the tree built to find merges
 *   (reusable for gravity only if anyMerged is false - a merge mutates the survivor's
 *   position/mass, which would make this tree stale for it), and whether anything
 *   actually merged this frame.
 */
export function mergeParticles(system, explosions) {
    const tree = buildQuadtree(system);
    const oldCount = system.count;

    // Upper bound on how far apart two particles could possibly be and still merge
    // (mergeThreshold = (radius_i + radius_j)/2), so the range query never misses a
    // legitimate candidate no matter which other particle it turns out to be.
    let globalMaxRadius = 0;
    for (let i = 0; i < oldCount; i++) {
        if (system.radius[i] > globalMaxRadius) globalMaxRadius = system.radius[i];
    }

    for (let i = 0; i < oldCount; i++) {
        system.removed[i] = 0;
    }

    const candidates = [];

    for (let i = 0; i < oldCount; i++) {
        if (system.removed[i]) {
            continue;
        }

        candidates.length = 0;
        findMergeCandidates(system, tree, i, (system.radius[i] + globalMaxRadius) / 2, candidates);

        for (const j of candidates) {
            if (system.removed[j]) continue;

            const dx = system.posX[j] - system.posX[i];
            const dy = system.posY[j] - system.posY[i];
            const distSq = dx * dx + dy * dy;
            const minDistance = (system.radius[i] + system.radius[j]) / 2;

            if (distSq > minDistance * minDistance) {
                continue; // not touching at all
            }

            // Touching isn't enough on its own - a slow graze (e.g. two bodies briefly
            // in contact while drifting past each other in similar orbits) just passes
            // through instead of fusing. Only a close approach with enough relative
            // (closing) speed actually merges.
            const relVx = system.velX[j] - system.velX[i];
            const relVy = system.velY[j] - system.velY[i];
            const relSpeedSq = relVx * relVx + relVy * relVy;

            if (relSpeedSq < constants.MIN_MERGE_VELOCITY * constants.MIN_MERGE_VELOCITY) {
                continue; // touching, but too gentle to fuse
            }

            mergeInto(system, i, j);
            system.removed[j] = 1;
            explosions.push(new Explosion(system.posX[i], system.posY[i], system.mass[i]));
            break; // one merge per particle per frame, same as before
        }
    }

    // Compacted AFTER every particle has had its turn, not removed incrementally during
    // the loop above: a particle that finds no match on its own turn can still be
    // absorbed later by a *different* particle's turn in the same pass (merge candidates
    // come from spatial proximity, not array order, so a particle with two valid
    // partners only merges with whichever is checked first - the other partner can still
    // come back for it on its own turn). Committing early left "ghosts": a particle
    // counted once on its own and again as part of whoever absorbed it later, duplicating
    // mass out of nowhere every time it happened.
    let writeIndex = 0;
    for (let i = 0; i < oldCount; i++) {
        if (system.removed[i]) continue;
        if (writeIndex !== i) {
            copyParticle(system, i, writeIndex);
        }
        writeIndex++;
    }
    const anyMerged = writeIndex !== oldCount;
    system.count = writeIndex;

    return { tree, anyMerged };
}
