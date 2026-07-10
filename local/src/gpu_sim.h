#pragma once
// Full-GPU physics pipeline (--gpu): integration, PM gravity (CIC deposit -> FFT Poisson
// solve -> gradient -> CIC gather), the P3M short-range correction, AND collision - all as
// OpenGL 4.3 compute shaders over SSBOs. Positions/velocities never leave GPU memory:
// renderer.h's renderFromBuffers draws directly from this pipeline's position and density
// buffers, so the only per-frame CPU<->GPU traffic is an 8-byte readback of max
// acceleration/speed (drives main.cpp's adaptive substepping and the neighbor-grid cell
// sizing - both intentionally one frame stale, the same lag the CPU path already has by
// using last frame's acceleration for this frame's substep decision).
//
// Numerical parity with the CPU path (pm_gravity.h / collide.h), and where it differs:
//  - The mesh solve is the same algorithm end-to-end (same CIC weights, same radix-2 FFT
//    math, same precomputed Green's table - uploaded from PMGrid::buildGreensTable - same
//    central-difference gradient), so the CPU-calibrated P3M table stays valid on GPU.
//  - Neighbor search uses a uniform grid instead of collide.h's tree: fixed-radius queries
//    are exactly what uniform grids are best at, and tree traversal (pointer-chasing,
//    divergent) is a poor fit for GPU warps. Cells overflowing MAX_PER_CELL drop the
//    excess (those pairs are missed for one substep) - graceful degradation, not UB.
//  - Collision resolution is Jacobi-style, not collide.h's sequential impulses: every
//    particle resolves against the same frozen begin-of-pass velocities and applies only
//    its own half of each pair's (symmetric, therefore still momentum-conserving) result.
//    A GPU can't replicate "each pair sees all earlier pairs' updated velocities" without
//    serializing - this is the standard reformulation, and adaptive substepping bounds the
//    error the same way it bounds integration error.
#include <GL/glew.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "particle_system.h"
#include "pm_gravity.h"

struct GpuSim {
    // Particle SSBOs
    GLuint posBuf = 0, velBuf = 0, accBuf = 0, propsBuf = 0; // props = vec4(invMass, radius, mass, 0) - invMass 0 encodes `fixed`
    GLuint velSnapBuf = 0;
    GLuint velDeltaBuf = 0; // vec4 per particle: xy = new velocity, zw = position correction
    GLuint densityOutBuf = 0; // per-particle 0..1 blur factor, consumed directly as a vertex attribute
    // Contact counts, ping-ponged between substeps: the collide pass reads LAST substep's
    // counts (frozen, so pair (i,j) and pair (j,i) both compute the identical
    // under-relaxation factor - see the collide shader) and writes this substep's.
    GLuint contactCountBufA = 0, contactCountBufB = 0;
    bool contactPing = false;
    // Mesh SSBOs
    GLuint densityGridBuf = 0; // uint, fixed-point atomics
    GLuint gridReA = 0, gridImA = 0, gridReB = 0, gridImB = 0; // A = natural layout, B = transposed scratch
    GLuint greensBuf = 0;      // Green's table, uploaded pre-transposed (it's applied in transposed layout)
    GLuint forceBuf = 0;       // vec2 per cell
    // Neighbor grid SSBOs (shared by collision and P3M)
    GLuint cellCountBuf = 0, cellItemsBuf = 0;
    // P3M calibration table
    GLuint p3mTableBuf = 0;
    // 2-uint readback: maxAccelSq bits, maxSpeedSq bits (float bits - monotonic for non-negatives)
    GLuint maxBitsBuf = 0;

    GLuint progIntegrate = 0, progBin = 0, progCollide = 0, progApply = 0, progDeposit = 0;
    GLuint progDens2Complex = 0, progFftRow = 0, progTranspose = 0, progGreens = 0;
    GLuint progGradient = 0, progGather = 0, progP3m = 0;

    int32_t count = 0;
    int32_t gridN = 0;
    float domainW = 0.f, domainH = 0.f;
    float cellW = 0.f, cellH = 0.f;
    float p3mRCut = 0.f;
    float maxParticleRadius = 0.f;
    // Neighbor grid covers 3x the window (escaped particles clamp into edge cells) - sized
    // at init for the smallest possible cell (= rCut), re-derived each frame as speeds grow.
    float binOriginX = 0.f, binOriginY = 0.f, binExtentW = 0.f, binExtentH = 0.f;
    int32_t maxBinCellsX = 0, maxBinCellsY = 0;
    // 64 (not higher): the neighbor grid's item store is maxCells * MAX_PER_CELL * 4 bytes,
    // and maxCells scales inversely with rCut^2 - at fine meshes (rCut ~5px over a 3x-window
    // extent) the store already reaches ~150MB at this setting. Overflowing cells drop the
    // excess (those pairs are missed for one substep) - graceful degradation.
    static constexpr int32_t MAX_PER_CELL = 64;
    static constexpr float DENSITY_FIXED_POINT_SCALE = 16777216.f; // 2^24 - see deposit shader comment

    float lastMaxAccel = 0.f, lastMaxSpeed = 0.f;

    static GLuint compileProgram(const std::string& src, const char* label) {
        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        const char* p = src.c_str();
        glShaderSource(shader, 1, &p, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[4096];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            fprintf(stderr, "Compute shader '%s' compile error:\n%s\n", label, log);
            exit(1);
        }
        GLuint prog = glCreateProgram();
        glAttachShader(prog, shader);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[4096];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            fprintf(stderr, "Compute program '%s' link error:\n%s\n", label, log);
            exit(1);
        }
        glDeleteShader(shader);
        return prog;
    }

    static GLuint makeSsbo(size_t bytes, const void* data = nullptr) {
        GLuint buf = 0;
        glGenBuffers(1, &buf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
        return buf;
    }

    static void clearUintBuffer(GLuint buf) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        GLuint zero = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    }

    void init(const ParticleSystem& sys, int32_t gridSize, float worldW, float worldH,
              PMGrid& cpuGrid, const PMShortRangeTable& table, float pmG, float pairSofteningFactor,
              float treeG, float treeSoftening, float restitution, float surfaceGap) {
        if (!GLEW_VERSION_4_3) {
            fprintf(stderr, "--gpu requires OpenGL 4.3 (compute shaders); this context doesn't provide it.\n");
            exit(1);
        }
        count = sys.count;
        gridN = gridSize;
        domainW = worldW;
        domainH = worldH;
        cellW = worldW / gridN;
        cellH = worldH / gridN;
        p3mRCut = table.rCut;

        // --- Particle buffers, uploaded once from the CPU-side spawn ---
        std::vector<float> pos2((size_t)count * 2), vel2((size_t)count * 2), props4((size_t)count * 4);
        maxParticleRadius = 0.f;
        for (int32_t i = 0; i < count; i++) {
            pos2[i * 2] = sys.posX[i];
            pos2[i * 2 + 1] = sys.posY[i];
            vel2[i * 2] = sys.velX[i];
            vel2[i * 2 + 1] = sys.velY[i];
            props4[i * 4] = sys.fixed[i] ? 0.f : 1.f / sys.mass[i];
            props4[i * 4 + 1] = sys.radius[i];
            props4[i * 4 + 2] = sys.mass[i];
            props4[i * 4 + 3] = 0.f;
            maxParticleRadius = std::max(maxParticleRadius, sys.radius[i]);
        }
        posBuf = makeSsbo(pos2.size() * 4, pos2.data());
        velBuf = makeSsbo(vel2.size() * 4, vel2.data());
        propsBuf = makeSsbo(props4.size() * 4, props4.data());
        accBuf = makeSsbo((size_t)count * 8);
        std::vector<float> zeros((size_t)count * 2, 0.f);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, accBuf);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, zeros.size() * 4, zeros.data());
        velSnapBuf = makeSsbo((size_t)count * 8);
        velDeltaBuf = makeSsbo((size_t)count * 16);
        densityOutBuf = makeSsbo((size_t)count * 4);
        contactCountBufA = makeSsbo((size_t)count * 4);
        contactCountBufB = makeSsbo((size_t)count * 4);
        clearUintBuffer(contactCountBufA);
        clearUintBuffer(contactCountBufB);

        // --- Mesh buffers ---
        size_t cells = (size_t)gridN * gridN;
        densityGridBuf = makeSsbo(cells * 4);
        gridReA = makeSsbo(cells * 4);
        gridImA = makeSsbo(cells * 4);
        gridReB = makeSsbo(cells * 4);
        gridImB = makeSsbo(cells * 4);
        forceBuf = makeSsbo(cells * 8);

        // Green's table, transposed at upload: the pipeline applies it between the two FFT
        // axis passes, i.e. while the grid sits in transposed layout (see substep()).
        cpuGrid.buildGreensTable(pmG);
        std::vector<float> greensT(cells);
        for (int32_t j = 0; j < gridN; j++) {
            for (int32_t i = 0; i < gridN; i++) {
                greensT[(size_t)i * gridN + j] = cpuGrid.greensScale[(size_t)j * gridN + i];
            }
        }
        greensBuf = makeSsbo(cells * 4, greensT.data());

        // --- Neighbor grid (worst case sizing: smallest cell = rCut) ---
        binExtentW = worldW * 3.f;
        binExtentH = worldH * 3.f;
        binOriginX = -worldW;
        binOriginY = -worldH;
        maxBinCellsX = (int32_t)ceilf(binExtentW / p3mRCut) + 1;
        maxBinCellsY = (int32_t)ceilf(binExtentH / p3mRCut) + 1;
        size_t maxCells = (size_t)maxBinCellsX * maxBinCellsY;
        cellCountBuf = makeSsbo(maxCells * 4);
        cellItemsBuf = makeSsbo(maxCells * MAX_PER_CELL * 4);

        p3mTableBuf = makeSsbo(table.accPerUnitMass.size() * 4, table.accPerUnitMass.data());
        maxBitsBuf = makeSsbo(8);

        buildPrograms(table.tableSize, pmG, pairSofteningFactor, treeG, treeSoftening, restitution, surfaceGap);
    }

    void buildPrograms(int32_t tableSize, float pmG, float pairSoft, float treeG, float treeSoft, float restitution, float gap) {
        char header[1024];
        snprintf(header, sizeof(header),
                 "#version 430\n"
                 "#define GRID_N %d\n"
                 "#define LOG2_N %d\n"
                 "#define HALF_N %d\n"
                 "#define MAX_PER_CELL %d\n"
                 "#define TABLE_SIZE %d\n"
                 "#define PM_G %.9g\n"
                 "#define PAIR_SOFT %.9g\n"
                 "#define TREE_G %.9g\n"
                 "#define TREE_SOFT %.9g\n"
                 "#define RESTITUTION %.9g\n"
                 "#define GAP %.9g\n"
                 "#define FP_SCALE %.1f\n",
                 gridN, (int32_t)log2f((float)gridN), gridN / 2, MAX_PER_CELL, tableSize,
                 pmG, pairSoft, treeG, treeSoft, restitution, gap, DENSITY_FIXED_POINT_SCALE);
        std::string h = header;

        progIntegrate = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Vel { vec2 vel[]; };
layout(std430, binding = 2) buffer Acc { vec2 acc[]; };
layout(std430, binding = 3) buffer Props { vec4 props[]; };
layout(std430, binding = 4) buffer MaxBits { uint maxAccelBits; uint maxSpeedBits; };
uniform float dt; uniform uint count;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    if (props[i].x != 0.0) { // invMass 0 = fixed particle, never moves
        vel[i] += acc[i] * dt;
        pos[i] += vel[i] * dt;
    }
    atomicMax(maxSpeedBits, floatBitsToUint(dot(vel[i], vel[i])));
}
)GLSL", "integrate");

        progBin = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer CellCount { uint cellCount[]; };
layout(std430, binding = 2) buffer CellItems { uint cellItems[]; };
uniform uint count; uniform vec2 binOrigin; uniform float cellSize; uniform ivec2 cellDims;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    ivec2 c = clamp(ivec2((pos[i] - binOrigin) / cellSize), ivec2(0), cellDims - 1);
    uint cell = uint(c.y * cellDims.x + c.x);
    uint slot = atomicAdd(cellCount[cell], 1u);
    if (slot < uint(MAX_PER_CELL)) cellItems[cell * uint(MAX_PER_CELL) + slot] = i;
    // slots past MAX_PER_CELL are dropped - those neighbors are missed for one substep
    // (graceful degradation; substepping revisits next step).
}
)GLSL", "bin");

        // Jacobi collision: same swept-contact + energy-accounted-overlap math as collide.h
        // (see that file for each formula's derivation), against FROZEN begin-of-pass
        // pos/velSnap, self-half application only, with symmetric under-relaxation (see
        // omega below). Also emits this substep's contact count (ping-ponged - next substep
        // reads it for omega) and the render blur factor.
        progCollide = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer VelSnap { vec2 velSnap[]; };
layout(std430, binding = 2) buffer Props { vec4 props[]; };
layout(std430, binding = 3) buffer CellCount { uint cellCount[]; };
layout(std430, binding = 4) buffer CellItems { uint cellItems[]; };
layout(std430, binding = 5) buffer VelDelta { vec4 velDelta[]; }; // xy = new vel, zw = pos correction
layout(std430, binding = 6) buffer PrevContacts { uint prevContacts[]; };
layout(std430, binding = 7) buffer NewContacts { uint newContacts[]; };
uniform uint count; uniform vec2 binOrigin; uniform float cellSize; uniform ivec2 cellDims;
uniform float dt; uniform float maxRadius;

float pairPE(float mi, float mj, float ri, float rj, float dist) {
    float combR = ri + rj;
    float sd = dist * dist + combR * combR * TREE_SOFT;
    return -(TREE_G * mi * mj) / sqrt(sd);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vec2 pi = pos[i];
    vec2 vi = velSnap[i];
    float invMi = props[i].x, ri = props[i].y, mi = props[i].z;
    float searchR = ri + maxRadius + GAP + length(vi);
    float searchR2 = searchR * searchR;

    // Pure Jacobi: every pair is evaluated from the FROZEN velSnap state on both sides, and
    // deltas only accumulate into dVel (never fed back into pair math within the pass). An
    // earlier version accumulated into a live myVel as the thread walked its neighbor list -
    // that made pair (i,j) computed from i's side see a different velocity than pair (j,i)
    // computed from j's side, breaking the equal-and-opposite symmetry and injecting net
    // momentum (observed directly as a steadily drifting center of mass). With frozen
    // inputs both sides compute the identical impulse, so momentum cancels exactly.
    //
    // Under-relaxation: each pair's impulse is scaled by 1/max(prevContacts_i,
    // prevContacts_j) - both read from LAST substep's frozen counts, so both sides compute
    // the identical factor (keeping the momentum symmetry above). Without this, a particle
    // overlapping k neighbors in a collapsed core receives k independently-computed
    // full-strength impulses per substep - where sequential CPU resolution lets each
    // contact see the previous one's result - and the sum pumps unbounded energy (measured
    // side by side vs the CPU path: GPU max speed hit ~600 at the moment of core formation
    // while CPU sat at ~5; with this factor the two track each other).
    vec2 dVel = vec2(0.0);
    vec2 myDelta = vec2(0.0);
    uint contacts = 0u;
    float myPrevContacts = float(max(prevContacts[i], 1u));

    ivec2 c0 = clamp(ivec2((pi - binOrigin) / cellSize) - 1, ivec2(0), cellDims - 1);
    ivec2 c1 = clamp(ivec2((pi - binOrigin) / cellSize) + 1, ivec2(0), cellDims - 1);
    for (int cy = c0.y; cy <= c1.y; cy++)
    for (int cx = c0.x; cx <= c1.x; cx++) {
        uint cell = uint(cy * cellDims.x + cx);
        uint n = min(cellCount[cell], uint(MAX_PER_CELL));
        for (uint s = 0u; s < n; s++) {
            uint j = cellItems[cell * uint(MAX_PER_CELL) + s];
            if (j == i) continue;
            vec2 pj = pos[j];
            vec2 endD = pj - pi;
            if (dot(endD, endD) > searchR2) continue;

            vec2 vj = velSnap[j];
            float invMj = props[j].x, rj = props[j].y, mj = props[j].z;
            if (invMi + invMj == 0.0) continue;

            vec2 relDrift = (vj - vi) * dt;
            vec2 startD = endD - relDrift;
            float touch = ri + rj + GAP;
            float cVal = dot(startD, startD) - touch * touch;
            float t; vec2 contact;
            bool contacted = true;
            if (cVal <= 0.0) {
                t = 0.0; contact = startD;
            } else {
                float a = dot(relDrift, relDrift);
                if (a < 1e-9) { contacted = false; t = 0.0; contact = startD; }
                else {
                    float b = 2.0 * dot(startD, relDrift);
                    float disc = b * b - 4.0 * a * cVal;
                    if (disc < 0.0) { contacted = false; t = 0.0; contact = startD; }
                    else {
                        t = (-b - sqrt(disc)) / (2.0 * a);
                        if (t < 0.0 || t > 1.0) { contacted = false; contact = startD; }
                        else contact = startD + t * relDrift;
                    }
                }
            }
            if (!contacted) continue;

            float omega = 1.0 / max(myPrevContacts, float(max(prevContacts[j], 1u)));

            float closing = dot(contact, vj - vi);
            bool anyContact = false;
            if (closing < 0.0) {
                // Approaching at contact: impulse bounce (collide.h resolveImpulse), self half.
                float cd = max(length(contact), 1e-6);
                vec2 nrm = contact / cd;
                float vn = dot(vi - vj, nrm);
                float imp = (1.0 + RESTITUTION) * vn / (invMi + invMj);
                vec2 dv = -imp * invMi * nrm * omega;
                dVel += dv;
                myDelta += (1.0 - t) * dt * dv;
                anyContact = true;
            }
            // Energy-accounted overlap separation (collide.h resolveOverlap), self half.
            float dist = max(length(endD), 1e-6);
            if (dist < touch) {
                vec2 nrm = endD / dist;
                float mu = 1.0 / (invMi + invMj);
                float vn = dot(vi - vj, nrm);
                float avail = 0.5 * mu * vn * vn;
                float fullCost = pairPE(mi, mj, ri, rj, touch) - pairPE(mi, mj, ri, rj, dist);
                float target;
                if (avail >= fullCost) target = touch;
                else {
                    float combR = ri + rj;
                    float softSq = combR * combR * TREE_SOFT;
                    float gm = TREE_G * mi * mj;
                    float k = 1.0 / sqrt(dist * dist + softSq) - avail / gm;
                    target = k <= 0.0 ? touch : sqrt(max(1.0 / (k * k) - softSq, 0.0));
                }
                if (target > dist) {
                    float wI = invMi / (invMi + invMj);
                    myDelta -= wI * (target - dist) * nrm * omega;
                    float spent = pairPE(mi, mj, ri, rj, target) - pairPE(mi, mj, ri, rj, dist);
                    float disc2 = max(vn * vn - 2.0 * spent / mu, 0.0);
                    float vnNew = sign(vn) * sqrt(disc2);
                    dVel -= mu * (vn - vnNew) * invMi * nrm * omega;
                }
                anyContact = true;
            }
            if (anyContact) contacts++;
        }
    }

    velDelta[i] = vec4(vi + dVel, myDelta);
    newContacts[i] = contacts;
}
)GLSL", "collide");

        progApply = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Vel { vec2 vel[]; };
layout(std430, binding = 2) buffer VelDelta { vec4 velDelta[]; };
layout(std430, binding = 3) buffer NewContacts { uint newContacts[]; };
layout(std430, binding = 4) buffer DensityOut { float densityOut[]; };
uniform uint count; uniform float densityBlurThreshold;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vel[i] = velDelta[i].xy;
    pos[i] += velDelta[i].zw;
    densityOut[i] = min(float(newContacts[i]) / densityBlurThreshold, 1.0);
}
)GLSL", "apply");

        // CIC scatter with fixed-point atomics (no portable float atomicAdd in core GL).
        // FP_SCALE = 2^24: max credible cell density (all mass in one cell ~= 27 mass/area
        // units) x 2^24 = ~4.6e8, comfortably under 2^31; a single smallest deposit at 1M
        // particles is still ~hundreds of integer units, so quantization noise is far below
        // the CIC interpolation error itself.
        progDeposit = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Props { vec4 props[]; };
layout(std430, binding = 2) buffer DensityGrid { uint densityGrid[]; };
uniform uint count; uniform vec2 cellSizes;
int wrapIdx(int v) { return ((v % GRID_N) + GRID_N) % GRID_N; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vec2 g = pos[i] / cellSizes;
    ivec2 i0 = ivec2(floor(g));
    vec2 f = g - vec2(i0);
    int x0 = wrapIdx(i0.x), y0 = wrapIdx(i0.y);
    int x1 = wrapIdx(i0.x + 1), y1 = wrapIdx(i0.y + 1);
    float m = props[i].z / (cellSizes.x * cellSizes.y);
    atomicAdd(densityGrid[y0 * GRID_N + x0], uint((1.0 - f.x) * (1.0 - f.y) * m * FP_SCALE));
    atomicAdd(densityGrid[y0 * GRID_N + x1], uint(f.x * (1.0 - f.y) * m * FP_SCALE));
    atomicAdd(densityGrid[y1 * GRID_N + x0], uint((1.0 - f.x) * f.y * m * FP_SCALE));
    atomicAdd(densityGrid[y1 * GRID_N + x1], uint(f.x * f.y * m * FP_SCALE));
}
)GLSL", "deposit");

        progDens2Complex = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer DensityGrid { uint densityGrid[]; };
layout(std430, binding = 1) buffer Re { float re[]; };
layout(std430, binding = 2) buffer Im { float im[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(GRID_N * GRID_N)) return;
    re[i] = float(densityGrid[i]) / FP_SCALE;
    im[i] = 0.0;
}
)GLSL", "dens2complex");

        // One workgroup per row, whole row in shared memory, radix-2 butterflies with
        // barriers between stages - the same math as fft.h's fft1d, laid out for a GPU.
        // outScale folds the inverse transform's 1/N-per-axis normalization into the store.
        progFftRow = compileProgram(h + R"GLSL(
layout(local_size_x = HALF_N) in;
layout(std430, binding = 0) buffer Re { float re[]; };
layout(std430, binding = 1) buffer Im { float im[]; };
uniform float dirSign;  // -1 forward, +1 inverse
uniform float outScale; // 1.0 forward, 1.0/GRID_N inverse
shared vec2 s[GRID_N];
void main() {
    uint row = gl_WorkGroupID.x;
    uint t = gl_LocalInvocationID.x;
    uint base = row * uint(GRID_N);
    for (uint e = t; e < uint(GRID_N); e += uint(HALF_N)) {
        uint r = bitfieldReverse(e) >> (32 - LOG2_N);
        s[r] = vec2(re[base + e], im[base + e]);
    }
    barrier();
    for (uint len = 2u; len <= uint(GRID_N); len <<= 1) {
        uint half_ = len >> 1;
        uint blk = t / half_, k = t % half_;
        uint i0 = blk * len + k, i1 = i0 + half_;
        float ang = dirSign * 6.28318530718 * float(k) / float(len);
        vec2 w = vec2(cos(ang), sin(ang));
        vec2 a = s[i0], b = s[i1];
        vec2 bw = vec2(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);
        s[i0] = a + bw;
        s[i1] = a - bw;
        barrier();
    }
    for (uint e = t; e < uint(GRID_N); e += uint(HALF_N)) {
        re[base + e] = s[e].x * outScale;
        im[base + e] = s[e].y * outScale;
    }
}
)GLSL", "fft_row");

        progTranspose = compileProgram(h + R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Src { float src[]; };
layout(std430, binding = 1) buffer Dst { float dst[]; };
shared float tile[16][17]; // +1 pad kills shared-memory bank conflicts
void main() {
    ivec2 g = ivec2(gl_WorkGroupID.xy) * 16 + ivec2(gl_LocalInvocationID.xy);
    if (g.x < GRID_N && g.y < GRID_N)
        tile[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = src[g.y * GRID_N + g.x];
    barrier();
    ivec2 gT = ivec2(gl_WorkGroupID.yx) * 16 + ivec2(gl_LocalInvocationID.xy);
    if (gT.x < GRID_N && gT.y < GRID_N)
        dst[gT.y * GRID_N + gT.x] = tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
}
)GLSL", "transpose");

        progGreens = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Re { float re[]; };
layout(std430, binding = 1) buffer Im { float im[]; };
layout(std430, binding = 2) buffer Greens { float greens[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(GRID_N * GRID_N)) return;
    re[i] *= greens[i];
    im[i] *= greens[i];
}
)GLSL", "greens");

        progGradient = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Phi { float phi[]; };
layout(std430, binding = 1) buffer Force { vec2 force[]; };
uniform vec2 cellSizes;
int wrapIdx(int v) { return ((v % GRID_N) + GRID_N) % GRID_N; }
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(GRID_N * GRID_N)) return;
    int x = int(idx) % GRID_N, y = int(idx) / GRID_N;
    float pxm = phi[y * GRID_N + wrapIdx(x - 1)];
    float pxp = phi[y * GRID_N + wrapIdx(x + 1)];
    float pym = phi[wrapIdx(y - 1) * GRID_N + x];
    float pyp = phi[wrapIdx(y + 1) * GRID_N + x];
    force[idx] = vec2(-(pxp - pxm) / (2.0 * cellSizes.x), -(pyp - pym) / (2.0 * cellSizes.y));
}
)GLSL", "gradient");

        progGather = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Force { vec2 force[]; };
layout(std430, binding = 2) buffer Acc { vec2 acc[]; };
uniform uint count; uniform vec2 cellSizes;
int wrapIdx(int v) { return ((v % GRID_N) + GRID_N) % GRID_N; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vec2 g = pos[i] / cellSizes;
    ivec2 i0 = ivec2(floor(g));
    vec2 f = g - vec2(i0);
    int x0 = wrapIdx(i0.x), y0 = wrapIdx(i0.y);
    int x1 = wrapIdx(i0.x + 1), y1 = wrapIdx(i0.y + 1);
    acc[i] = (1.0 - f.x) * (1.0 - f.y) * force[y0 * GRID_N + x0]
           + f.x * (1.0 - f.y) * force[y0 * GRID_N + x1]
           + (1.0 - f.x) * f.y * force[y1 * GRID_N + x0]
           + f.x * f.y * force[y1 * GRID_N + x1];
}
)GLSL", "gather");

        // Same correction formula + linear table interpolation as pm_gravity.h's
        // applyP3MCorrection (write-only-to-self, so no atomics on acc).
        progP3m = compileProgram(h + R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Props { vec4 props[]; };
layout(std430, binding = 2) buffer Acc { vec2 acc[]; };
layout(std430, binding = 3) buffer CellCount { uint cellCount[]; };
layout(std430, binding = 4) buffer CellItems { uint cellItems[]; };
layout(std430, binding = 5) buffer Table { float meshTable[]; };
layout(std430, binding = 6) buffer MaxBits { uint maxAccelBits; uint maxSpeedBits; };
uniform uint count; uniform vec2 binOrigin; uniform float cellSize; uniform ivec2 cellDims;
uniform float rCut;
float tableLookup(float r) {
    float p = r / rCut * float(TABLE_SIZE) - 0.5;
    if (p <= 0.0) return meshTable[0];
    int k = int(p);
    if (k >= TABLE_SIZE - 1) return meshTable[TABLE_SIZE - 1];
    float fr = p - float(k);
    return meshTable[k] * (1.0 - fr) + meshTable[k + 1] * fr;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vec2 pi = pos[i];
    float ri = props[i].y;
    vec2 sum = vec2(0.0);
    ivec2 c0 = clamp(ivec2((pi - binOrigin) / cellSize) - 1, ivec2(0), cellDims - 1);
    ivec2 c1 = clamp(ivec2((pi - binOrigin) / cellSize) + 1, ivec2(0), cellDims - 1);
    for (int cy = c0.y; cy <= c1.y; cy++)
    for (int cx = c0.x; cx <= c1.x; cx++) {
        uint cell = uint(cy * cellDims.x + cx);
        uint n = min(cellCount[cell], uint(MAX_PER_CELL));
        for (uint s = 0u; s < n; s++) {
            uint j = cellItems[cell * uint(MAX_PER_CELL) + s];
            if (j == i) continue;
            vec2 d = pos[j] - pi;
            float r = length(d);
            if (r <= 0.0 || r >= rCut) continue;
            float combR = ri + props[j].y;
            float softenedR = sqrt(r * r + combR * combR * PAIR_SOFT);
            float mj = props[j].z;
            float corr = 2.0 * PM_G * mj / softenedR - mj * tableLookup(r);
            sum += corr * d / r;
        }
    }
    vec2 a = acc[i] + sum;
    acc[i] = a;
    atomicMax(maxAccelBits, floatBitsToUint(dot(a, a)));
}
)GLSL", "p3m");
    }

    static int32_t groups(int32_t n, int32_t local) { return (n + local - 1) / local; }

    // One kick-drift-collide-gravity cycle, all on GPU - mirrors main.cpp's CPU substep
    // structure dispatch for dispatch.
    void substep(float dt, float densityBlurThreshold, float treeGap) {
        // Neighbor cell size: big enough for both collision's speed-padded reach and P3M's
        // rCut. Uses last frame's max speed (x1.5 margin) - one frame stale by design.
        float reach = 2.f * maxParticleRadius + treeGap + lastMaxSpeed * 1.5f + 1.f;
        float cellSize = std::max(p3mRCut, reach);
        int32_t dimsX = std::min((int32_t)ceilf(binExtentW / cellSize) + 1, maxBinCellsX);
        int32_t dimsY = std::min((int32_t)ceilf(binExtentH / cellSize) + 1, maxBinCellsY);

        int32_t pg = groups(count, 256);
        int32_t cg = groups(gridN * gridN, 256);

        // 1. kick + drift
        glUseProgram(progIntegrate);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, accBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, propsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, maxBitsBuf);
        glUniform1f(glGetUniformLocation(progIntegrate, "dt"), dt);
        glUniform1ui(glGetUniformLocation(progIntegrate, "count"), count);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        // 2. freeze velocities for the Jacobi collision pass
        glBindBuffer(GL_COPY_READ_BUFFER, velBuf);
        glBindBuffer(GL_COPY_WRITE_BUFFER, velSnapBuf);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (size_t)count * 8);

        // 3. clear per-substep grids
        clearUintBuffer(cellCountBuf);
        clearUintBuffer(densityGridBuf);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 4. bin particles into the neighbor grid
        glUseProgram(progBin);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellCountBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cellItemsBuf);
        glUniform1ui(glGetUniformLocation(progBin, "count"), count);
        glUniform2f(glGetUniformLocation(progBin, "binOrigin"), binOriginX, binOriginY);
        glUniform1f(glGetUniformLocation(progBin, "cellSize"), cellSize);
        glUniform2i(glGetUniformLocation(progBin, "cellDims"), dimsX, dimsY);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 5. collide (Jacobi, writes velDelta + this substep's contact counts; reads last
        // substep's counts for the symmetric under-relaxation factor)
        GLuint prevContacts = contactPing ? contactCountBufB : contactCountBufA;
        GLuint newContacts = contactPing ? contactCountBufA : contactCountBufB;
        contactPing = !contactPing;
        glUseProgram(progCollide);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velSnapBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, propsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cellCountBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellItemsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, velDeltaBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, prevContacts);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, newContacts);
        glUniform1ui(glGetUniformLocation(progCollide, "count"), count);
        glUniform2f(glGetUniformLocation(progCollide, "binOrigin"), binOriginX, binOriginY);
        glUniform1f(glGetUniformLocation(progCollide, "cellSize"), cellSize);
        glUniform2i(glGetUniformLocation(progCollide, "cellDims"), dimsX, dimsY);
        glUniform1f(glGetUniformLocation(progCollide, "dt"), dt);
        glUniform1f(glGetUniformLocation(progCollide, "maxRadius"), maxParticleRadius);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 6. apply collision results (+ density blur factor from contact counts)
        glUseProgram(progApply);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, velDeltaBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, newContacts);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, densityOutBuf);
        glUniform1ui(glGetUniformLocation(progApply, "count"), count);
        glUniform1f(glGetUniformLocation(progApply, "densityBlurThreshold"), densityBlurThreshold);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 7. CIC deposit from corrected positions
        glUseProgram(progDeposit);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, propsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, densityGridBuf);
        glUniform1ui(glGetUniformLocation(progDeposit, "count"), count);
        glUniform2f(glGetUniformLocation(progDeposit, "cellSizes"), cellW, cellH);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 8. fixed-point density -> complex field
        glUseProgram(progDens2Complex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, densityGridBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridReA);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridImA);
        glDispatchCompute(cg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 9-15. FFT Poisson solve: rowFFT -> transpose -> rowFFT -> greens (in transposed
        // layout, table pre-transposed at init) -> inverse rowFFT -> transpose -> inverse
        // rowFFT, with the 1/N-per-axis inverse scaling folded into the inverse passes.
        auto fftPass = [&](GLuint reB, GLuint imB, float sign, float scale) {
            glUseProgram(progFftRow);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, reB);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, imB);
            glUniform1f(glGetUniformLocation(progFftRow, "dirSign"), sign);
            glUniform1f(glGetUniformLocation(progFftRow, "outScale"), scale);
            glDispatchCompute(gridN, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        auto transposePass = [&](GLuint src, GLuint dst) {
            glUseProgram(progTranspose);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dst);
            glDispatchCompute(groups(gridN, 16), groups(gridN, 16), 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };

        float invN = 1.f / (float)gridN;
        fftPass(gridReA, gridImA, -1.f, 1.f);
        transposePass(gridReA, gridReB);
        transposePass(gridImA, gridImB);
        fftPass(gridReB, gridImB, -1.f, 1.f);

        glUseProgram(progGreens);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gridReB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridImB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, greensBuf);
        glDispatchCompute(cg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        fftPass(gridReB, gridImB, 1.f, invN);
        transposePass(gridReB, gridReA);
        transposePass(gridImB, gridImA);
        fftPass(gridReA, gridImA, 1.f, invN);

        // 16. force field = -grad(phi)
        glUseProgram(progGradient);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gridReA);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, forceBuf);
        glUniform2f(glGetUniformLocation(progGradient, "cellSizes"), cellW, cellH);
        glDispatchCompute(cg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 17. CIC gather -> acc
        glUseProgram(progGather);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, forceBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, accBuf);
        glUniform1ui(glGetUniformLocation(progGather, "count"), count);
        glUniform2f(glGetUniformLocation(progGather, "cellSizes"), cellW, cellH);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 18. P3M short-range correction onto acc (+ records max |acc|^2 for substepping)
        glUseProgram(progP3m);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, propsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, accBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cellCountBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellItemsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, p3mTableBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, maxBitsBuf);
        glUniform1ui(glGetUniformLocation(progP3m, "count"), count);
        glUniform2f(glGetUniformLocation(progP3m, "binOrigin"), binOriginX, binOriginY);
        glUniform1f(glGetUniformLocation(progP3m, "cellSize"), cellSize);
        glUniform2i(glGetUniformLocation(progP3m, "cellDims"), dimsX, dimsY);
        glUniform1f(glGetUniformLocation(progP3m, "rCut"), p3mRCut);
        glDispatchCompute(pg, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // End-of-frame: read back this frame's max |acc| and |vel| (8 bytes - the pipeline's
    // only recurring GPU->CPU transfer) and reset the accumulators for next frame.
    void finishFrame() {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        uint32_t bits[2] = {0, 0};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBitsBuf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8, bits);
        float accSq, speedSq;
        memcpy(&accSq, &bits[0], 4);
        memcpy(&speedSq, &bits[1], 4);
        lastMaxAccel = sqrtf(std::max(accSq, 0.f));
        lastMaxSpeed = sqrtf(std::max(speedSq, 0.f));
        clearUintBuffer(maxBitsBuf);
    }

    // Occasional diagnostics readback (debug panel COM, once per second) - not on the
    // per-frame hot path.
    void readPositions(std::vector<float>& outXY) {
        outXY.resize((size_t)count * 2);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, posBuf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, outXY.size() * 4, outXY.data());
    }
};
