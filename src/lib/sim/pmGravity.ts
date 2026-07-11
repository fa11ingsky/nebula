// Particle-Mesh (PM) gravity - a direct port of the native app's local/src/pm_gravity.h,
// selectable as an alternative gravity solver to gravity.ts's Barnes-Hut tree. The classic
// Hockney & Eastwood PM steps:
//   1. Deposit particle mass onto a grid (Cloud-in-Cell: each particle spreads its mass
//      across the 4 nearest grid points, bilinear-weighted).
//   2. Solve the Poisson equation on the grid via FFT: transform the density, multiply by
//      the Green's function for the Laplacian in Fourier space, transform back.
//   3. Difference the potential into a force field (central differences per axis).
//   4. Interpolate that force back onto each particle using the SAME CIC weights used for
//      deposition - that symmetry is what avoids a particle exerting a net self-force.
//
// IMPORTANT: this simulation is 2D, so solving d^2(phi) = 4*pi*G*rho directly on a 2D grid
// gives genuine 2D gravity - a logarithmic potential and a ~1/r force law - not the
// 3D-style ~1/r^2 law the tree path uses on these same 2D positions. PM is a different
// gravity *model*, not a faster version of the same physics, which is why it has its own
// separately-tuned gravitational constant (constants.PM_GRAVITATIONAL_CONSTANT).
//
// The mesh alone can't resolve structure below ~cell size - measured in the native app as
// a genuine force error at close range (~125x too strong at 20px on a 256-grid), not just
// "softer" gravity. The P3M ("Particle-Particle-Particle-Mesh") short-range correction
// (PMShortRangeTable, applyP3MCorrection) fixes close pairs by replacing the mesh's wrong
// contribution with a correctly-softened direct pairwise force.
//
// BOUNDARY CONDITIONS: the mesh solve uses ISOLATED (free-space) boundaries via the
// classic Hockney & Eastwood zero-padding trick - FFT grids are 2x the physical
// resolution per axis, and the density is convolved with the true free-space 2D kernel
// (a softened G*ln(r^2) potential, built in real space with min-image displacements and
// FFT'd once) instead of a periodic Green's function. A plain FFT Poisson solve is
// inherently periodic - it tiles the domain to infinity, so every particle is also pulled
// by phantom image copies of the whole swarm one domain away. On a typical wide canvas
// those images sit much closer vertically than horizontally, which measurably weakened
// the inward pull on top/bottom particles vs side particles (2.7x at the spawn disc's
// edge, verified in the native app with a ring-of-probes test) - visible at sim start as
// the sides collapsing inward faster. The padding costs 4x the FFT area; deposit,
// gradient, and gather run on the padded grid too, so particles up to a full domain-width
// outside the window keep correct forces (only farther ones alias back around).
import { fft2d } from './fft.ts';

// Gaussian low-pass length for the mesh's own Green's function, in units of sqrt(cell
// area) - paired deliberately with PM_P3M_CUTOFF_FACTOR (see constants.ts): a softer mesh
// needs a longer P3M reach before it can be trusted alone, and P3M cost scales with reach
// squared. Same value as the native app's PM_MESH_SOFTENING_CELLS.
export const PM_MESH_SOFTENING_CELLS = 1.5;

export function createPMGrid(nx, ny, width, height) {
    // fftNx/fftNy are the zero-padded FFT dims (2x per axis) for isolated boundaries -
    // see this file's header. All work arrays live on the padded grid.
    const fftNx = 2 * nx, fftNy = 2 * ny;
    const fftTotal = fftNx * fftNy;
    return {
        nx, ny,
        fftNx, fftNy,
        domainWidth: width,
        domainHeight: height,
        cellWidth: width / nx,
        cellHeight: height / ny,
        density: new Float32Array(fftTotal),
        potentialRe: new Float32Array(fftTotal),
        potentialIm: new Float32Array(fftTotal),
        forceX: new Float32Array(fftTotal),
        forceY: new Float32Array(fftTotal),
        // Free-space kernel spectrum - fully static for a given grid/G. Real-valued: the
        // real-space kernel is even under min-image negation, so its DFT is real (the
        // float-noise imaginary part is discarded at build time).
        kernelHat: new Float32Array(fftTotal),
        kernelBuiltForG: 0,
    };
}

function wrap(i, n) {
    i %= n;
    return i < 0 ? i + n : i;
}

/**
 * Isolated (free-space) kernel: the 2D Poisson equation d^2(phi) = 4*pi*G*rho has the
 * free-space point response phi(r) = 2*G*ln(r) (d^2(ln r) = 2*pi*delta in 2D),
 * Plummer-softened here to G*ln(r^2 + eps^2) with eps = the PM_MESH_SOFTENING_CELLS mesh
 * softening. Built in REAL space on the doubled grid with min-image displacements, then
 * FFT'd once - multiplying the padded density's spectrum by this and inverting IS the
 * discrete free-space convolution, valid for all source/target separations up to one
 * full physical domain (half the doubled grid). cellArea is folded in because the
 * density grid stores mass/cellArea while the convolution needs plain mass.
 */
export function buildIsolatedKernelTable(grid, G) {
    const eps = PM_MESH_SOFTENING_CELLS * Math.sqrt(grid.cellWidth * grid.cellHeight);
    const epsSq = eps * eps;
    const cellArea = grid.cellWidth * grid.cellHeight;
    const kIm = new Float32Array(grid.fftNx * grid.fftNy);
    for (let j = 0; j < grid.fftNy; j++) {
        const dy = j <= grid.fftNy / 2 ? j : j - grid.fftNy;
        const y = dy * grid.cellHeight;
        for (let i = 0; i < grid.fftNx; i++) {
            const dx = i <= grid.fftNx / 2 ? i : i - grid.fftNx;
            const x = dx * grid.cellWidth;
            grid.kernelHat[j * grid.fftNx + i] = G * Math.log(x * x + y * y + epsSq) * cellArea;
        }
    }
    fft2d(grid.kernelHat, kIm, grid.fftNx, grid.fftNy, false);
    // kernelHat now holds the (real) spectrum; kIm is float noise and is discarded.
    grid.kernelBuiltForG = G;
}

/**
 * The mesh-only PM solve - deposit, FFT, Green's function, force-diff, interpolate.
 * Zeroes accX/accY itself before accumulating (the native version's confirmed
 * accumulate-without-reset bug is why this is explicit). CIC weights are stashed in
 * weightsScratch (a Float32Array of [ix, iy, fx, fy] quads, grown as needed) so the
 * gather step reuses exactly the deposit's weights.
 */
export function computeGravityPMMesh(system, grid, G, scratch) {
    const count = system.count;
    if (grid.kernelBuiltForG !== G) {
        buildIsolatedKernelTable(grid, G);
    }

    for (let i = 0; i < count; i++) {
        system.accX[i] = 0;
        system.accY[i] = 0;
    }
    grid.density.fill(0);

    if (scratch.weights.length < count * 4) {
        scratch.weights = new Float32Array(count * 4);
    }
    const weights = scratch.weights;
    // Deposit/gather indices wrap on the PADDED grid: a particle slightly outside the
    // window lands in pad cells at its true offset (correct under the min-image
    // free-space kernel out to a full domain-width away).
    const fnx = grid.fftNx, fny = grid.fftNy;
    const cellArea = grid.cellWidth * grid.cellHeight;

    for (let i = 0; i < count; i++) {
        const gx = system.posX[i] / grid.cellWidth;
        const gy = system.posY[i] / grid.cellHeight;
        const ix = Math.floor(gx);
        const iy = Math.floor(gy);
        const fx = gx - ix;
        const fy = gy - iy;
        weights[i * 4] = ix;
        weights[i * 4 + 1] = iy;
        weights[i * 4 + 2] = fx;
        weights[i * 4 + 3] = fy;

        const ix0 = wrap(ix, fnx), iy0 = wrap(iy, fny);
        const ix1 = wrap(ix + 1, fnx), iy1 = wrap(iy + 1, fny);
        const m = system.mass[i] / cellArea;
        grid.density[iy0 * fnx + ix0] += (1 - fx) * (1 - fy) * m;
        grid.density[iy0 * fnx + ix1] += fx * (1 - fy) * m;
        grid.density[iy1 * fnx + ix0] += (1 - fx) * fy * m;
        grid.density[iy1 * fnx + ix1] += fx * fy * m;
    }

    grid.potentialIm.fill(0);
    grid.potentialRe.set(grid.density);
    fft2d(grid.potentialRe, grid.potentialIm, fnx, fny, false);

    // Isolated-boundary Poisson solve by convolution: multiply the padded density's
    // spectrum by the free-space kernel spectrum and invert (see buildIsolatedKernelTable
    // and this file's header).
    const total = fnx * fny;
    for (let idx = 0; idx < total; idx++) {
        grid.potentialRe[idx] *= grid.kernelHat[idx];
        grid.potentialIm[idx] *= grid.kernelHat[idx];
    }

    fft2d(grid.potentialRe, grid.potentialIm, fnx, fny, true);

    // Force = -grad(phi), central differences over the whole padded grid (wrapping on
    // the padded dims is still correct at its edges: cell -1 is cell fftNx-1 by min-image).
    for (let j = 0; j < fny; j++) {
        const jm = wrap(j - 1, fny), jp = wrap(j + 1, fny);
        for (let i = 0; i < fnx; i++) {
            const im_ = wrap(i - 1, fnx), ip = wrap(i + 1, fnx);
            const idx = j * fnx + i;
            grid.forceX[idx] = -(grid.potentialRe[j * fnx + ip] - grid.potentialRe[j * fnx + im_]) / (2 * grid.cellWidth);
            grid.forceY[idx] = -(grid.potentialRe[jp * fnx + i] - grid.potentialRe[jm * fnx + i]) / (2 * grid.cellHeight);
        }
    }

    for (let i = 0; i < count; i++) {
        const ix = weights[i * 4], iy = weights[i * 4 + 1];
        const fx = weights[i * 4 + 2], fy = weights[i * 4 + 3];
        const ix0 = wrap(ix, fnx), iy0 = wrap(iy, fny);
        const ix1 = wrap(ix + 1, fnx), iy1 = wrap(iy + 1, fny);
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
        const w01 = (1 - fx) * fy, w11 = fx * fy;
        system.accX[i] += w00 * grid.forceX[iy0 * fnx + ix0] + w10 * grid.forceX[iy0 * fnx + ix1]
                        + w01 * grid.forceX[iy1 * fnx + ix0] + w11 * grid.forceX[iy1 * fnx + ix1];
        system.accY[i] += w00 * grid.forceY[iy0 * fnx + ix0] + w10 * grid.forceY[iy0 * fnx + ix1]
                        + w01 * grid.forceY[iy1 * fnx + ix0] + w11 * grid.forceY[iy1 * fnx + ix1];
    }
}

/** Reusable scratch for computeGravityPMMesh / computeGravityPM. */
export function createPMScratch() {
    return {
        weights: new Float32Array(0),
        // Uniform-grid neighbor search scratch for applyP3MCorrection (see below).
        cellCount: new Int32Array(0),
        cellStart: new Int32Array(0),
        cellItems: new Int32Array(0),
        particleCell: new Int32Array(0),
    };
}

/**
 * P3M short-range correction table. Calibration runs the actual mesh solve on an isolated
 * massless-test-particle + unit-mass-source pair at tableSize separations spanning
 * (0, rCut], so the table is automatically self-consistent with whatever the mesh solve
 * actually does (its softening, FFT numerics, CIC interpolation). Runs on a small
 * FIXED-resolution calibration grid whose physical CELL SIZE matches production - the
 * short-range mesh response being tabulated is set by cell size, and calibrating on the
 * full production grid would make startup scale with grid area (the native app measured
 * minutes of FFTs at grid 2048). 64: the isolated-boundary solve pads its FFTs to 2x per
 * axis (quadrupling per-entry cost) and is image-free at any domain size that simply
 * contains rCut (~4 cells), so the roomier domain the periodic version needed is now
 * pure waste - 64 keeps the JS calibration to a few hundred ms.
 */
const CAL_GRID_N = 64;

export function calibratePMShortRangeTable(cellW, cellH, G, rCut, tableSize) {
    const table = {
        accPerUnitMass: new Float32Array(tableSize),
        rCut,
        tableSize,
    };

    const calDomainW = cellW * CAL_GRID_N;
    const calDomainH = cellH * CAL_GRID_N;
    const calGrid = createPMGrid(CAL_GRID_N, CAL_GRID_N, calDomainW, calDomainH);
    const scratch = createPMScratch();
    const centerX = calDomainW * 0.5, centerY = calDomainH * 0.5;

    // A minimal 2-particle stand-in for the real particle system - only the fields
    // computeGravityPMMesh reads.
    const calSys = {
        count: 2,
        posX: new Float32Array(2),
        posY: new Float32Array(2),
        mass: new Float32Array([0, 1]), // massless test particle + unit-mass source
        accX: new Float32Array(2),
        accY: new Float32Array(2),
    };

    for (let k = 0; k < tableSize; k++) {
        const r = rCut * (k + 1) / tableSize;
        calSys.posX[0] = centerX - r * 0.5;
        calSys.posY[0] = centerY;
        calSys.posX[1] = centerX + r * 0.5;
        calSys.posY[1] = centerY;
        computeGravityPMMesh(calSys, calGrid, G, scratch);
        // Test particle sits left of the source, along +x - a correctly attractive solve
        // gives accX[0] > 0. This is exactly meshAccPerUnitSourceMass(r).
        table.accPerUnitMass[k] = calSys.accX[0];
    }
    return table;
}

function tableLookup(table, r) {
    if (table.tableSize === 0 || r >= table.rCut) return 0;
    const pos = r / table.rCut * table.tableSize - 0.5;
    if (pos <= 0) return table.accPerUnitMass[0];
    const k0 = Math.floor(pos);
    if (k0 >= table.tableSize - 1) return table.accPerUnitMass[table.tableSize - 1];
    const frac = pos - k0;
    return table.accPerUnitMass[k0] * (1 - frac) + table.accPerUnitMass[k0 + 1] * frac;
}

/**
 * Finds pairs within table.rCut and replaces the mesh's already-applied (wrong, at this
 * range) contribution with the correct direct pairwise force. Neighbor search is a flat
 * uniform grid (counting-sort layout: count pass, prefix sum, fill pass) rather than the
 * native version's tree - fixed-radius queries are exactly what uniform grids are best
 * at, and this keeps pmGravity.ts self-contained. Write-only-to-self accumulation, same
 * as the native version: each particle sums corrections from every neighbor onto its own
 * acceleration only, visiting each pair twice with exactly equal-and-opposite results.
 */
export function applyP3MCorrection(system, table, G, pairSofteningFactor, scratch) {
    if (table.tableSize === 0) return;
    const count = system.count;
    const rCut = table.rCut;

    // Grid extent from the particles' actual bounding box - escaped particles just make
    // cells sparser, they don't break anything.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < count; i++) {
        if (system.posX[i] < minX) minX = system.posX[i];
        if (system.posX[i] > maxX) maxX = system.posX[i];
        if (system.posY[i] < minY) minY = system.posY[i];
        if (system.posY[i] > maxY) maxY = system.posY[i];
    }
    const cellsX = Math.max(1, Math.min(2048, Math.ceil((maxX - minX) / rCut) + 1));
    const cellsY = Math.max(1, Math.min(2048, Math.ceil((maxY - minY) / rCut) + 1));
    const cellW = Math.max(rCut, (maxX - minX) / cellsX);
    const cellH = Math.max(rCut, (maxY - minY) / cellsY);
    const totalCells = cellsX * cellsY;

    if (scratch.cellCount.length < totalCells) {
        scratch.cellCount = new Int32Array(totalCells);
        scratch.cellStart = new Int32Array(totalCells + 1);
    }
    if (scratch.cellItems.length < count) {
        scratch.cellItems = new Int32Array(count);
        scratch.particleCell = new Int32Array(count);
    }
    const cellCount = scratch.cellCount, cellStart = scratch.cellStart;
    const cellItems = scratch.cellItems, particleCell = scratch.particleCell;
    cellCount.fill(0, 0, totalCells);

    for (let i = 0; i < count; i++) {
        const cx = Math.min(cellsX - 1, Math.max(0, Math.floor((system.posX[i] - minX) / cellW)));
        const cy = Math.min(cellsY - 1, Math.max(0, Math.floor((system.posY[i] - minY) / cellH)));
        const cell = cy * cellsX + cx;
        particleCell[i] = cell;
        cellCount[cell]++;
    }
    let running = 0;
    for (let c = 0; c < totalCells; c++) {
        cellStart[c] = running;
        running += cellCount[c];
        cellCount[c] = 0; // reused as the fill cursor below
    }
    cellStart[totalCells] = running;
    for (let i = 0; i < count; i++) {
        const cell = particleCell[i];
        cellItems[cellStart[cell] + cellCount[cell]++] = i;
    }

    const rCutSq = rCut * rCut;
    for (let i = 0; i < count; i++) {
        const px = system.posX[i], py = system.posY[i];
        const ri = system.radius[i];
        const ccx = particleCell[i] % cellsX;
        const ccy = (particleCell[i] - ccx) / cellsX;
        let sumX = 0, sumY = 0;
        const cy0 = Math.max(0, ccy - 1), cy1 = Math.min(cellsY - 1, ccy + 1);
        const cx0 = Math.max(0, ccx - 1), cx1 = Math.min(cellsX - 1, ccx + 1);
        for (let cy = cy0; cy <= cy1; cy++) {
            for (let cx = cx0; cx <= cx1; cx++) {
                const cell = cy * cellsX + cx;
                const end = cellStart[cell + 1];
                for (let s = cellStart[cell]; s < end; s++) {
                    const j = cellItems[s];
                    if (j === i) continue;
                    const dx = system.posX[j] - px;
                    const dy = system.posY[j] - py;
                    const rSq = dx * dx + dy * dy;
                    if (rSq <= 0 || rSq >= rCutSq) continue;
                    const r = Math.sqrt(rSq);
                    const invR = 1 / r;

                    const combinedRadius = ri + system.radius[j];
                    const softenedR = Math.sqrt(rSq + combinedRadius * combinedRadius * pairSofteningFactor);
                    const meshTerm = tableLookup(table, r);

                    // 2D PM force law: the mesh solves a true 2D Poisson equation, so the
                    // correct short-range pair force is ~1/r (from a log potential), i.e.
                    // 2*G*m/softenedR - matching the native applyP3MCorrection exactly.
                    const correction = 2 * G * system.mass[j] / softenedR - system.mass[j] * meshTerm;
                    sumX += correction * dx * invR;
                    sumY += correction * dy * invR;
                }
            }
        }
        system.accX[i] += sumX;
        system.accY[i] += sumY;
    }
}

/**
 * Full PM gravity pass: mesh solve (long-range and aggregate multi-body gravity) plus the
 * P3M short-range correction (close pairs, where the mesh alone is wrong). This is what
 * the worker's PM mode actually calls each frame.
 */
export function computeGravityPM(system, grid, table, G, pairSofteningFactor, scratch) {
    computeGravityPMMesh(system, grid, G, scratch);
    applyP3MCorrection(system, table, G, pairSofteningFactor, scratch);
}
