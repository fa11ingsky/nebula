#pragma once
// Particle-Mesh (PM) gravity - an alternative to tree.h/gravity.h's Barnes-Hut tree,
// selected via --pm on the command line (see main.cpp). The classic Hockney & Eastwood PM
// steps:
//   1. Deposit particle mass onto a grid (Cloud-in-Cell: each particle spreads its mass
//      across the 4 nearest grid points, bilinear-weighted - smoother than Nearest-Grid-
//      Point, which gives forces that are discontinuous in value as a particle crosses a
//      cell boundary).
//   2. Solve the Poisson equation on the grid via FFT: transform the density, multiply by
//      the Green's function for the Laplacian in Fourier space, transform back to get the
//      potential.
//   3. Difference the potential into a force field (central differences per axis).
//   4. Interpolate that force back onto each particle using the SAME CIC weights used for
//      deposition in step 1 - that symmetry (not just using CIC at all) is what avoids a
//      particle exerting a net self-force and keeps momentum close to conserved, the same
//      property gravity.h's tree traversal gets from a different mechanism (every particle
//      independently computing an equal-and-opposite direct-pair force).
//
// Cost is independent of particle clustering - unlike the tree, which visits more nodes per
// particle as density rises (see this port's own profiling investigations under a dense
// central-mass cluster), a fixed-size grid costs the same O(gridN^2 log gridN) FFT
// regardless of how the particles are distributed. The tradeoff is resolution: the mesh
// alone can't resolve structure below ~cell size - measured directly (an isolated 2-particle
// test, no collision, no other forces) as a genuine, non-conservative force error at close
// range (~125x too strong at 20px on a 256-grid), not just "softer" gravity. Steps 1-4 above
// (computeGravityPMMesh) handle everything the mesh CAN resolve correctly; a P3M
// ("Particle-Particle-Particle-Mesh") short-range correction pass (PMShortRangeTable,
// applyP3MCorrection) fixes close pairs by replacing the mesh's wrong contribution with a
// correctly-softened direct pairwise force - see those for the full derivation.
// computeGravityPM (bottom of this file) is the combined entry point main.cpp actually calls.
//
// IMPORTANT: this simulation is 2D, so solving d^2(phi) = 4*pi*G*rho directly on a 2D grid
// gives genuine 2D gravity - a logarithmic potential and a ~1/r force law - not the
// 3D-style ~1/r^2 law gravity.h's tree path uses on these same 2D positions. --pm is a
// different gravity *model*, not a drop-in faster version of the same physics; two
// particles pulling on each other will behave differently under one path than the other,
// especially at long range. Reproducing an actual 1/r^2 law from a PM solve would need a
// modified ("razor-thin disk") Green's function or a 3D grid - out of scope here.
//
// BOUNDARY CONDITIONS: the CPU mesh solve uses ISOLATED (free-space/vacuum) boundaries via
// the classic Hockney & Eastwood zero-padding trick - the FFT grids are 2x the physical
// resolution per axis, and the density is convolved with the true free-space 2D kernel
// (a softened G*ln(r^2) potential, built in real space with min-image displacements and
// FFT'd once) instead of the periodic -4*pi*G/k^2 Green's function. A plain (unpadded)
// FFT Poisson solve is inherently periodic: it implicitly tiles the domain to infinity,
// so every particle is also attracted by phantom image copies of the whole swarm one
// domain-width away. On this app's 1600x900 window those images sit only 900px away
// vertically vs 1600px horizontally, which measurably (2.7x at the spawn disc's edge,
// confirmed with a ring-of-test-particles probe) weakened the inward pull on top/bottom
// particles relative to side particles - visible at sim start as the sides collapsing
// inward faster. The tree path never had this because direct summation is open-boundary
// by construction. The padding costs 4x the FFT area; deposit/gather/gradient run on the
// padded grid too (a particle slightly outside the window lands in pad cells at its true
// offset - only particles more than a full domain away alias back in).
//
// NOTE: the GPU pipeline (gpu_sim.h) still runs the PERIODIC solve - its shared-memory
// FFT is built around rows of exactly GRID_N and doubling it would overflow workgroup
// shared memory at large grids. It uses PMGrid::buildGreensTable (the periodic k-space
// table, kept for exactly that purpose); the CPU path uses buildIsolatedKernelTable.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>
#include "particle_system.h"
#include "tree.h"
#include "fft.h"

constexpr float PM_PI = 3.14159265358979323846f;

// Gaussian low-pass length for the mesh's own Green's function, in units of sqrt(cell
// area) - see the Green's-table comment in computeGravityPMMesh. Paired deliberately with
// constants.h's PM_P3M_CUTOFF_FACTOR: a softer mesh needs a longer P3M reach before it can
// be trusted alone (and P3M pair count scales with reach *squared* - the original 3.0
// softening + 6-cell cutoff combination measured 437ms/solve at 50k particles, nearly all
// of it P3M pairs). Sharpening the mesh here lets the cutoff shrink; whatever short-range
// error the sharper mesh has inside the cutoff is exactly what the P3M calibration table
// cancels anyway, so this trades cost, not correctness.
constexpr float PM_MESH_SOFTENING_CELLS = 1.5f;

struct PMGrid {
    int32_t nx = 0, ny = 0;         // physical resolution (what callers configure)
    int32_t fftNx = 0, fftNy = 0;   // 2*nx, 2*ny - zero-padded for isolated boundaries (see file header)
    float domainWidth = 0.f, domainHeight = 0.f;
    float cellWidth = 0.f, cellHeight = 0.f;

    // All sized fftNx*fftNy: deposit/gradient/gather run on the padded grid so particles
    // slightly outside the window keep interacting at their true offsets (see file header).
    std::vector<float> density;
    std::vector<float> potentialRe, potentialIm;
    std::vector<float> forceX, forceY;

    // PERIODIC k-space Green's table, nx*ny - used ONLY by the GPU pipeline (gpu_sim.h),
    // which still runs the periodic solve; see the file header. Fully static for a given
    // grid/domain/G, so recomputing the expf per cell per frame (as an earlier version
    // did) was pure waste, measured at ~1.4ms/solve on a 256 grid.
    std::vector<float> greensScale;
    float greensBuiltForG = 0.f; // 0 = not built; rebuilt automatically if G changes

    // ISOLATED free-space kernel spectrum, fftNx*fftNy - the CPU solve's multiplier.
    // Real-valued: the real-space kernel is even under min-image negation, so its DFT is
    // real (up to float noise, which is dropped along with the whole imaginary part).
    std::vector<float> kernelHat;
    float kernelBuiltForG = 0.f;

    void allocate(int32_t gridNx, int32_t gridNy, float width, float height) {
        nx = gridNx;
        ny = gridNy;
        fftNx = 2 * gridNx;
        fftNy = 2 * gridNy;
        domainWidth = width;
        domainHeight = height;
        cellWidth = width / (float)nx;
        cellHeight = height / (float)ny;
        size_t fftTotal = (size_t)fftNx * fftNy;
        density.assign(fftTotal, 0.f);
        potentialRe.assign(fftTotal, 0.f);
        potentialIm.assign(fftTotal, 0.f);
        forceX.assign(fftTotal, 0.f);
        forceY.assign(fftTotal, 0.f);
        greensScale.assign((size_t)nx * ny, 0.f);
        greensBuiltForG = 0.f;
        kernelHat.assign(fftTotal, 0.f);
        kernelBuiltForG = 0.f;
    }

    // Periodic k-space table - GPU pipeline only (see the struct comment above).
    void buildGreensTable(float G) {
        float softeningLength = PM_MESH_SOFTENING_CELLS * sqrtf(cellWidth * cellHeight);
        float softeningLengthSq = softeningLength * softeningLength;
        for (int32_t j = 0; j < ny; j++) {
            int32_t ky = j <= ny / 2 ? j : j - ny;
            float kyReal = 2.f * PM_PI * (float)ky / domainHeight;
            for (int32_t i = 0; i < nx; i++) {
                int32_t kx = i <= nx / 2 ? i : i - nx;
                float kxReal = 2.f * PM_PI * (float)kx / domainWidth;
                float k2 = kxReal * kxReal + kyReal * kyReal;
                // k=0 (the mean mode) gets scale 0: it's an overall constant offset to phi,
                // physically meaningless (only gradients matter) and singular in the formula.
                greensScale[(size_t)j * nx + i] = k2 == 0.f ? 0.f : (-4.f * PM_PI * G / k2) * expf(-k2 * softeningLengthSq);
            }
        }
        greensBuiltForG = G;
    }

    // Isolated (free-space) kernel for the CPU solve: the 2D Poisson equation
    // d^2(phi) = 4*pi*G*rho has the free-space point response phi(r) = 2*G*ln(r)
    // (since d^2(ln r) = 2*pi*delta in 2D), Plummer-softened here to G*ln(r^2 + eps^2)
    // with eps = the same PM_MESH_SOFTENING_CELLS mesh softening the periodic table used.
    // Built in REAL space on the doubled grid with min-image displacements (cell (i,j)
    // holds the kernel at physical offset (min-image(i)*cellW, min-image(j)*cellH)), then
    // FFT'd once - multiplying the padded density's spectrum by this and inverting IS the
    // discrete free-space convolution, valid for all source/target separations up to one
    // full physical domain (half the doubled grid). cellArea is folded in because the
    // density grid stores mass/cellArea while the convolution needs plain mass.
    void buildIsolatedKernelTable(float G, int32_t maxThreads = 1) {
        float eps = PM_MESH_SOFTENING_CELLS * sqrtf(cellWidth * cellHeight);
        float epsSq = eps * eps;
        float cellArea = cellWidth * cellHeight;
        std::vector<float> kIm((size_t)fftNx * fftNy, 0.f);
        for (int32_t j = 0; j < fftNy; j++) {
            int32_t dy = j <= fftNy / 2 ? j : j - fftNy;
            float y = (float)dy * cellHeight;
            for (int32_t i = 0; i < fftNx; i++) {
                int32_t dx = i <= fftNx / 2 ? i : i - fftNx;
                float x = (float)dx * cellWidth;
                kernelHat[(size_t)j * fftNx + i] = G * logf(x * x + y * y + epsSq) * cellArea;
            }
        }
        fft2d(kernelHat.data(), kIm.data(), fftNx, fftNy, false, maxThreads);
        // kernelHat now holds the (real) spectrum; kIm is float noise and is discarded.
        kernelBuiltForG = G;
    }

    inline int32_t wrap(int32_t i, int32_t n) const {
        i %= n;
        return i < 0 ? i + n : i;
    }
};

// (ix, iy, fx, fy) for one particle's Cloud-in-Cell footprint - computed once at deposit
// time and reused unchanged at interpolation time (see this file's header comment on why
// that symmetry matters), rather than recomputed from position (which could subtly drift
// if anything moved the particle between the two steps - it doesn't here, but storing the
// exact weights removes any doubt).
struct CicWeights {
    int32_t ix, iy;
    float fx, fy;
};

// Indices wrap on the PADDED (fftNx/fftNy) grid: a particle slightly outside the window
// deposits into pad cells at its true relative offset - correct under the min-image
// free-space kernel out to a full domain-width away. Only particles farther out than
// that alias back around (they were teleporting a full window earlier under the old
// periodic solve anyway - this strictly widens the correct zone).
inline CicWeights depositCIC(PMGrid& grid, float x, float y, float mass) {
    float gx = x / grid.cellWidth;
    float gy = y / grid.cellHeight;
    int32_t ix = (int32_t)floorf(gx);
    int32_t iy = (int32_t)floorf(gy);
    float fx = gx - (float)ix;
    float fy = gy - (float)iy;

    int32_t ix0 = grid.wrap(ix, grid.fftNx);
    int32_t iy0 = grid.wrap(iy, grid.fftNy);
    int32_t ix1 = grid.wrap(ix + 1, grid.fftNx);
    int32_t iy1 = grid.wrap(iy + 1, grid.fftNy);

    float cellArea = grid.cellWidth * grid.cellHeight;
    float w00 = (1.f - fx) * (1.f - fy) * mass / cellArea;
    float w10 = fx * (1.f - fy) * mass / cellArea;
    float w01 = (1.f - fx) * fy * mass / cellArea;
    float w11 = fx * fy * mass / cellArea;

    grid.density[(size_t)iy0 * grid.fftNx + ix0] += w00;
    grid.density[(size_t)iy0 * grid.fftNx + ix1] += w10;
    grid.density[(size_t)iy1 * grid.fftNx + ix0] += w01;
    grid.density[(size_t)iy1 * grid.fftNx + ix1] += w11;

    return {ix, iy, fx, fy};
}

inline void interpolateForceCIC(const PMGrid& grid, const CicWeights& w, float& outAx, float& outAy) {
    int32_t ix0 = grid.wrap(w.ix, grid.fftNx);
    int32_t iy0 = grid.wrap(w.iy, grid.fftNy);
    int32_t ix1 = grid.wrap(w.ix + 1, grid.fftNx);
    int32_t iy1 = grid.wrap(w.iy + 1, grid.fftNy);

    float w00 = (1.f - w.fx) * (1.f - w.fy);
    float w10 = w.fx * (1.f - w.fy);
    float w01 = (1.f - w.fx) * w.fy;
    float w11 = w.fx * w.fy;

    outAx = w00 * grid.forceX[(size_t)iy0 * grid.fftNx + ix0]
          + w10 * grid.forceX[(size_t)iy0 * grid.fftNx + ix1]
          + w01 * grid.forceX[(size_t)iy1 * grid.fftNx + ix0]
          + w11 * grid.forceX[(size_t)iy1 * grid.fftNx + ix1];
    outAy = w00 * grid.forceY[(size_t)iy0 * grid.fftNx + ix0]
          + w10 * grid.forceY[(size_t)iy0 * grid.fftNx + ix1]
          + w01 * grid.forceY[(size_t)iy1 * grid.fftNx + ix0]
          + w11 * grid.forceY[(size_t)iy1 * grid.fftNx + ix1];
}

// The mesh-only PM solve - deposit, FFT, Green's function, force-diff, interpolate. Zeroes
// sys.accX/accY itself before accumulating into them - matching gravity.h's computeGravity,
// which does the same (see its own tree-traversal reset) - rather than trusting the caller
// to have zeroed them first. THIS WAS A REAL, CONFIRMED BUG until fixed here: main.cpp never
// calls resetAccelerationAll itself, so with only a += here (no reset), acceleration
// silently accumulated without bound across every single frame regardless of actual
// separation or force - the true root cause of particles "not attracting, flying off
// randomly" under --pm. Traced via an isolated calibration comparison that explicitly did
// reset acceleration (see PMShortRangeTable::calibrate) landing exactly on the correct,
// expected force magnitude, while the real per-frame simulation loop (missing the reset)
// did not - the mesh solve itself was correct at long range the whole time.
//
// Kept separate from computeGravityPM (below) so PMShortRangeTable::calibrate can call this
// core directly, without the P3M correction it exists to calibrate against.
struct PMTimings {
    double depositMs = 0, fftFwdMs = 0, greensMs = 0, fftInvMs = 0, gradMs = 0, interpMs = 0, p3mMs = 0;
};

inline void computeGravityPMMesh(ParticleSystem& sys, PMGrid& grid, float G, std::vector<CicWeights>& weightsScratch, PMTimings* timings = nullptr, int32_t maxThreads = 1) {
    int32_t count = sys.count;
    auto stageStart = std::chrono::steady_clock::now();
    auto lap = [&](double* out) {
        if (!out) return;
        auto now = std::chrono::steady_clock::now();
        *out += std::chrono::duration<double, std::milli>(now - stageStart).count();
        stageStart = now;
    };

    if (grid.kernelBuiltForG != G) {
        grid.buildIsolatedKernelTable(G, maxThreads);
    }

    for (int32_t i = 0; i < sys.count; i++) {
        sys.accX[i] = 0.f;
        sys.accY[i] = 0.f;
    }
    std::fill(grid.density.begin(), grid.density.end(), 0.f);

    weightsScratch.resize(count);
    for (int32_t i = 0; i < count; i++) {
        weightsScratch[i] = depositCIC(grid, sys.posX[i], sys.posY[i], sys.mass[i]);
    }
    lap(timings ? &timings->depositMs : nullptr);

    std::fill(grid.potentialIm.begin(), grid.potentialIm.end(), 0.f);
    std::copy(grid.density.begin(), grid.density.end(), grid.potentialRe.begin());

    fft2d(grid.potentialRe.data(), grid.potentialIm.data(), grid.fftNx, grid.fftNy, false, maxThreads);
    lap(timings ? &timings->fftFwdMs : nullptr);

    // Isolated-boundary Poisson solve by convolution: multiply the padded density's
    // spectrum by the precomputed free-space kernel spectrum (see
    // buildIsolatedKernelTable) and invert. Because the physical mass occupies only one
    // quadrant of the doubled grid and the kernel uses min-image displacements, this is
    // an exact discrete free-space convolution - no periodic images (see file header for
    // the measured artifact those caused).
    {
        size_t total = (size_t)grid.fftNx * grid.fftNy;
        const float* scale = grid.kernelHat.data();
        for (size_t idx = 0; idx < total; idx++) {
            grid.potentialRe[idx] *= scale[idx];
            grid.potentialIm[idx] *= scale[idx];
        }
    }
    lap(timings ? &timings->greensMs : nullptr);

    fft2d(grid.potentialRe.data(), grid.potentialIm.data(), grid.fftNx, grid.fftNy, true, maxThreads);
    lap(timings ? &timings->fftInvMs : nullptr);
    // potentialIm should now be ~0 (up to float noise) since the input density was purely
    // real and the kernel spectrum is purely real - only potentialRe is used below.

    // Force = -grad(phi), central differences over the whole padded grid (particles just
    // outside the window sit in pad cells - see depositCIC). Wrapping indices on the
    // padded grid is still correct at its edges: cell -1 is cell fftNx-1 by min-image.
    for (int32_t j = 0; j < grid.fftNy; j++) {
        int32_t jm = grid.wrap(j - 1, grid.fftNy);
        int32_t jp = grid.wrap(j + 1, grid.fftNy);
        for (int32_t i = 0; i < grid.fftNx; i++) {
            int32_t im = grid.wrap(i - 1, grid.fftNx);
            int32_t ip = grid.wrap(i + 1, grid.fftNx);
            float phiXm = grid.potentialRe[(size_t)j * grid.fftNx + im];
            float phiXp = grid.potentialRe[(size_t)j * grid.fftNx + ip];
            float phiYm = grid.potentialRe[(size_t)jm * grid.fftNx + i];
            float phiYp = grid.potentialRe[(size_t)jp * grid.fftNx + i];
            size_t idx = (size_t)j * grid.fftNx + i;
            grid.forceX[idx] = -(phiXp - phiXm) / (2.f * grid.cellWidth);
            grid.forceY[idx] = -(phiYp - phiYm) / (2.f * grid.cellHeight);
        }
    }
    lap(timings ? &timings->gradMs : nullptr);

    for (int32_t i = 0; i < count; i++) {
        float ax, ay;
        interpolateForceCIC(grid, weightsScratch[i], ax, ay);
        sys.accX[i] += ax;
        sys.accY[i] += ay;
    }
    lap(timings ? &timings->interpMs : nullptr);
}

// P3M short-range correction (Particle-Particle-Particle-Mesh) - the actual fix for the
// mesh's confirmed close-range force error, not just a mitigation. Root cause, confirmed
// with an isolated 2-particle test (no collision, no other forces): at separation ~20px on
// a 256-grid (cellWidth~6.25px), computeGravityPMMesh's interpolated acceleration was
// ~125x stronger than the well-resolved far-field 1/r falloff predicts - sustained over
// enough frames to pump real, permanent extra kinetic energy into a pair instead of a clean
// elastic swing-by. Gaussian softening (above) reduces this but can't eliminate it without
// also suppressing genuine short-range gravity everywhere else - the actual fix has to
// treat close pairs differently from the mesh's own (wrong, at that range) answer.
//
// The idea: for any pair closer than rCut, SUBTRACT what the mesh solve *already,
// incorrectly* contributed for that pair (estimated via a calibration table - see below) and
// ADD BACK the correct, directly-computed pairwise force instead. Pairs farther than rCut
// are left alone (the mesh is trusted to be accurate there - confirmed by the original
// isolated-pair test's far-field measurements). This is the textbook P3M split: mesh handles
// long-range (and multi-body aggregate) gravity everywhere, direct pairwise summation
// handles short-range for the (usually small) set of nearby pairs where the mesh can't
// resolve fine enough.
struct PMShortRangeTable {
    std::vector<float> accPerUnitMass;
    float rCut = 0.f;
    int32_t tableSize = 0;

    // Table entry k (0-indexed) is sampled at r = (k+1)/tableSize * rCut - i.e. tableSize
    // samples spanning (0, rCut], never exactly at r=0 (undefined/singular for a bare point
    // mass; the *correction* formula in applyP3MCorrection has its own separate softening
    // for that regime, using the pair's actual combined radius, not this table).
    //
    // Calibration works by actually running computeGravityPMMesh on an isolated pair - not
    // an analytic approximation, so it's automatically self-consistent with whatever the
    // mesh solve actually does (including its own softening, FFT numerics, CIC interpolation
    // - anything else that'd be hard to derive a closed form for). The test particle carries
    // zero mass, so its own deposit contributes nothing to the density field - isolating
    // "how much does the *other* particle's mass affect force here" from any self-force
    // contamination (a separate, smaller, well-known PM artifact this isn't targeting).
    //
    // Runs on a small FIXED-resolution grid whose physical CELL SIZE matches production
    // (domain scaled down to cellSize * CAL_GRID_N per axis), not the production grid
    // itself: the short-range mesh response being tabulated is set by cell size - rCut is
    // only ~8 cells, far inside the calibration domain - while calibrating on the full
    // production grid made startup scale with grid area (measured: MINUTES of CPU FFTs at
    // grid 2048 x 128 table entries). 128 (was 256): the isolated-boundary solve pads its
    // FFTs to 2x per axis, quadrupling per-entry cost, and also ELIMINATED the one reason
    // a roomy calibration domain mattered (periodic images contaminating the measured
    // response) - the isolated solve is image-free at any domain size that simply
    // contains rCut, which 128 cells does many times over.
    static constexpr int32_t CAL_GRID_N = 128;
    void calibrate(float cellW, float cellH, float G, float rCutIn, int32_t tableSizeIn) {
        rCut = rCutIn;
        tableSize = tableSizeIn;
        accPerUnitMass.assign(tableSize, 0.f);

        float calDomainW = cellW * CAL_GRID_N;
        float calDomainH = cellH * CAL_GRID_N;
        PMGrid calGrid;
        calGrid.allocate(CAL_GRID_N, CAL_GRID_N, calDomainW, calDomainH);
        std::vector<CicWeights> calWeights;

        ParticleSystem calSys;
        calSys.allocate(4);
        float centerX = calDomainW * 0.5f, centerY = calDomainH * 0.5f;

        for (int32_t k = 0; k < tableSize; k++) {
            float r = rCut * (float)(k + 1) / (float)tableSize;
            calSys.count = 0;
            addParticle(calSys, centerX - r * 0.5f, centerY, 0.f, 1.f, 1.f, 1.f); // massless test particle
            addParticle(calSys, centerX + r * 0.5f, centerY, 1.f, 1.f, 1.f, 1.f); // unit-mass source
            computeGravityPMMesh(calSys, calGrid, G, calWeights, nullptr, 4); // zeroes acceleration itself; threads the padded FFTs
            // Test particle sits left of the source, along +x - a correctly attractive
            // solve gives accX[0] > 0 (pulled toward the source). This is exactly
            // meshAccPerUnitSourceMass(r), since the source carries unit mass.
            accPerUnitMass[k] = calSys.accX[0];
        }
    }

    // Linear interpolation between table samples; 0 beyond rCut (mesh presumed accurate -
    // no correction to make) or if never calibrated.
    inline float lookup(float r) const {
        if (tableSize == 0 || r >= rCut) return 0.f;
        float pos = r / rCut * (float)tableSize - 0.5f; // table[k] lives at pos=k
        if (pos <= 0.f) return accPerUnitMass[0];
        int32_t k0 = (int32_t)pos;
        if (k0 >= tableSize - 1) return accPerUnitMass[tableSize - 1];
        float frac = pos - (float)k0;
        return accPerUnitMass[k0] * (1.f - frac) + accPerUnitMass[k0 + 1] * frac;
    }
};

// Finds pairs within table.rCut via a dedicated tree (fresh-built from current positions -
// see main.cpp's p3mTree) and replaces the mesh's already-applied (wrong, at this range)
// contribution with the correct direct pairwise force.
//
// Threaded with the write-only-to-self formulation: each particle i sums corrections from
// every neighbor j onto its OWN accX[i]/accY[i] only, never j's - so a contiguous split of
// the particle range across threads has no shared writes at all (same race-free-by-
// construction shape as gravity.h's traversal threading). This visits each pair twice
// (once from each side) instead of once with symmetric application - double the pair math,
// but it removes the write race entirely, and both sides compute exactly equal-and-opposite
// results by construction, so momentum behavior is identical to the old sequential
// j>i version. Net measured win at 50k particles: far more than 2x back from threading.
inline void applyP3MCorrection(ParticleSystem& sys, SpatialTree& p3mTree, const PMShortRangeTable& table, float G, float pairSofteningFactor, int32_t maxThreads = 1) {
    if (table.tableSize == 0) return;
    int32_t count = sys.count;

    p3mTree.leafCapacity = 4; // fine leaves - exact small-radius queries, same reasoning as collide.h's own tree
    p3mTree.maxDepth = 18;
    p3mTree.build(sys, count);

    auto correctRange = [&](int32_t startI, int32_t endI) {
        constexpr int32_t MAX_P3M_CANDIDATES = 256;
        int32_t candBuf[MAX_P3M_CANDIDATES]; // per-thread - lives on each worker's own stack
        for (int32_t i = startI; i < endI; i++) {
            int32_t found = p3mTree.findNearbyInto(sys, i, table.rCut, candBuf, MAX_P3M_CANDIDATES);
            float sumX = 0.f, sumY = 0.f;
            for (int32_t c = 0; c < found; c++) {
                int32_t j = candBuf[c];
                float dx = sys.posX[j] - sys.posX[i];
                float dy = sys.posY[j] - sys.posY[i];
                float r = sqrtf(dx * dx + dy * dy);
                if (r <= 0.f || r >= table.rCut) continue;
                float invR = 1.f / r;

                float combinedRadius = sys.radius[i] + sys.radius[j];
                float softenedR = sqrtf(r * r + combinedRadius * combinedRadius * pairSofteningFactor);
                float meshTerm = table.lookup(r);

                // Acceleration on i from j's mass - independent of the *receiving*
                // particle's own mass (equivalence principle), same as gravity.h's
                // applyDirectGravity.
                float correction = 2.f * G * sys.mass[j] / softenedR - sys.mass[j] * meshTerm;
                sumX += correction * dx * invR;
                sumY += correction * dy * invR;
            }
            sys.accX[i] += sumX;
            sys.accY[i] += sumY;
        }
    };

    unsigned int hw = std::thread::hardware_concurrency();
    int32_t threadCount = hw < 1 ? 1 : (int32_t)std::min(hw, (unsigned int)std::max(maxThreads, 1));
    if (threadCount > count) threadCount = count > 0 ? count : 1;

    if (threadCount <= 1) {
        correctRange(0, count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        int32_t chunk = (count + threadCount - 1) / threadCount;
        for (int32_t t = 0; t < threadCount; t++) {
            int32_t startI = t * chunk;
            int32_t endI = std::min(startI + chunk, count);
            if (startI >= endI) break;
            workers.emplace_back(correctRange, startI, endI);
        }
        for (auto& w : workers) w.join();
    }
}

// Full PM gravity pass: mesh solve (long-range and aggregate multi-body gravity) plus the
// P3M short-range correction (close pairs, where the mesh alone is wrong - see
// applyP3MCorrection's header comment). This is the function main.cpp actually calls.
inline void computeGravityPM(ParticleSystem& sys, PMGrid& grid, SpatialTree& p3mTree, const PMShortRangeTable& table,
                              float G, float pairSofteningFactor, std::vector<CicWeights>& weightsScratch,
                              PMTimings* timings = nullptr, int32_t maxThreads = 1) {
    computeGravityPMMesh(sys, grid, G, weightsScratch, timings, maxThreads);
    auto p3mStart = std::chrono::steady_clock::now();
    applyP3MCorrection(sys, p3mTree, table, G, pairSofteningFactor, maxThreads);
    if (timings) {
        timings->p3mMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p3mStart).count();
    }
}
