// Uniform spatial grid for collide.ts's "find nearby particles" queries - deliberately
// separate from quadtree.ts's Barnes-Hut tree (used for gravity), since collision only
// ever needs particles within a bounded, roughly-constant-scale radius (particle size),
// which a flat grid answers via a handful of direct cell lookups - versus the tree's
// O(log n) recursive descent with node-by-node pruning tests, which pays for hierarchy
// the collision query never needed in the first place. In collision (non-merging) mode,
// where the swarm never shrinks and the quadtree used to get built twice a frame (once
// for collision's own neighbor search, once more for gravity, since gravity can't safely
// reuse a tree collision may have perturbed), replacing collision's copy with this grid
// leaves exactly one quadtree build per frame instead of two.
//
// Built fresh every frame (positions change every frame) via a counting-sort bucket
// layout: every particle's cell id is computed once, a prefix sum turns per-cell counts
// into start offsets, then a second pass drops each particle's index into its cell's
// contiguous slice of one flat array - no per-cell array allocations, no hash map.
import constants from '../constants.ts';
import { computeCenterOfMass } from './particleSystem.ts';

// Reused across every build instead of allocating fresh typed arrays 60 times a second -
// the same reuse-and-grow pattern the quadtree's node store uses.
let cellSize = 1;
let minX = 0;
let minY = 0;
let gridWidth = 0;
let gridHeight = 0;
let cellStart = new Int32Array(1); // length gridWidth*gridHeight+1
let sortedIndices = new Int32Array(0); // length = particle count, grouped by cell
let cellOf = new Int32Array(0); // length >= particle count; each particle's cell id

function ensureParticleCapacity(count) {
    if (sortedIndices.length < count) {
        sortedIndices = new Int32Array(count);
    }
    if (cellOf.length < count) {
        cellOf = new Int32Array(count);
    }
}

function ensureCellCapacity(cellCount) {
    if (cellStart.length < cellCount + 1) {
        cellStart = new Int32Array(cellCount + 1);
    } else {
        cellStart.fill(0, 0, cellCount + 1);
    }
}

/**
 * Builds a fresh grid covering the current particle positions. Cell size is set to the
 * largest possible touch distance in the system (2x the biggest radius, plus the
 * cosmetic surface gap) - the standard "cell size >= interaction radius" invariant that
 * guarantees any two touching particles land in the same or an immediately adjacent
 * cell, so the bounded neighborhood search in findNearbyInGrid (widened further only for
 * particles moving fast enough this frame to need it) never misses a real collision.
 *
 * Sized off the RMS spread from the center of mass, same rationale as quadtree.ts's
 * buildQuadtree - keeps one far-flung outlier from blowing up the grid to cover empty
 * space nobody else occupies. Particles further out are clamped to the grid boundary for
 * cell-assignment purposes only; their true position still drives every actual distance
 * check.
 *
 * @returns an opaque grid handle - pass it into findNearbyInGrid.
 */
export function buildSpatialGrid(system) {
    const count = system.count;
    ensureParticleCapacity(count);

    let globalMaxRadius = 0;
    for (let i = 0; i < count; i++) {
        if (system.radius[i] > globalMaxRadius) globalMaxRadius = system.radius[i];
    }
    cellSize = Math.max(2 * globalMaxRadius + constants.COLLISION_SURFACE_GAP, 1e-3);

    const com = computeCenterOfMass(system);
    let sumSqDeviation = 0;
    for (let i = 0; i < count; i++) {
        const dx = system.posX[i] - com.x;
        const dy = system.posY[i] - com.y;
        sumSqDeviation += dx * dx + dy * dy;
    }
    const rmsSpread = count > 0 ? Math.sqrt(sumSqDeviation / count) : 1;
    const halfSize = Math.max(rmsSpread * 6, cellSize * 4);

    minX = com.x - halfSize;
    minY = com.y - halfSize;
    gridWidth = Math.max(Math.ceil((halfSize * 2) / cellSize), 1);
    gridHeight = gridWidth; // square region, matching the quadtree's own square root node

    const maxCellCoord = gridWidth - 1;
    const cellCount = gridWidth * gridHeight;
    ensureCellCapacity(cellCount);

    // Pass 1: compute each particle's cell id, tallying counts into cellStart shifted up
    // by one slot (cellStart[c+1] ends up holding cell c's count, ready for the prefix
    // sum below to turn into a start offset).
    for (let i = 0; i < count; i++) {
        const cx = Math.min(Math.max(Math.floor((system.posX[i] - minX) / cellSize), 0), maxCellCoord);
        const cy = Math.min(Math.max(Math.floor((system.posY[i] - minY) / cellSize), 0), maxCellCoord);
        const cellId = cy * gridWidth + cx;
        cellOf[i] = cellId;
        cellStart[cellId + 1]++;
    }

    for (let c = 0; c < cellCount; c++) {
        cellStart[c + 1] += cellStart[c];
    }

    // Pass 2: drop each particle into its cell's slice. Reusing cellStart's own values as
    // write cursors would clobber the start offsets pass 1 just computed (still needed
    // for every query afterward), so a scratch copy tracks the next free slot per cell.
    const cursor = cellStart.slice(0, cellCount);
    for (let i = 0; i < count; i++) {
        const cellId = cellOf[i];
        sortedIndices[cursor[cellId]] = i;
        cursor[cellId]++;
    }

    return { cellSize, minX, minY, gridWidth, gridHeight, cellStart, sortedIndices };
}

/**
 * Collects every particle index within searchRadius of particle i into `out` - the
 * grid-based counterpart to quadtree.ts's findNearbyParticles. Checks every cell within
 * the Chebyshev-distance neighborhood needed to guarantee covering searchRadius, usually
 * just the immediate 3x3 block, wider only for a particle moving fast enough this frame
 * to need it (see collide.ts's own per-particle search radius, unchanged from before).
 *
 * Matches findNearbyParticles's contract exactly: candidates are a superset of the true
 * answer (no exact-distance filter here, since collide.ts's own swept-collision math
 * already has to do a precise check on every candidate anyway - filtering twice would
 * just be wasted work), never a subset.
 */
export function findNearbyInGrid(grid, system, i, searchRadius, out) {
    const px = system.posX[i];
    const py = system.posY[i];

    const cellRadius = Math.max(Math.ceil(searchRadius / grid.cellSize), 1);
    const centerCx = Math.min(Math.max(Math.floor((px - grid.minX) / grid.cellSize), 0), grid.gridWidth - 1);
    const centerCy = Math.min(Math.max(Math.floor((py - grid.minY) / grid.cellSize), 0), grid.gridHeight - 1);

    const minCx = Math.max(centerCx - cellRadius, 0);
    const maxCx = Math.min(centerCx + cellRadius, grid.gridWidth - 1);
    const minCy = Math.max(centerCy - cellRadius, 0);
    const maxCy = Math.min(centerCy + cellRadius, grid.gridHeight - 1);

    for (let cy = minCy; cy <= maxCy; cy++) {
        const rowBase = cy * grid.gridWidth;
        for (let cx = minCx; cx <= maxCx; cx++) {
            const cellId = rowBase + cx;
            const start = grid.cellStart[cellId];
            const end = grid.cellStart[cellId + 1];
            for (let k = start; k < end; k++) {
                const j = grid.sortedIndices[k];
                if (j !== i) out.push(j);
            }
        }
    }
}
