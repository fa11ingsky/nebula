// Full-GPU physics pipeline - a WebGPU/WGSL port of the native app's local/src/gpu_sim.h:
// integration, PM gravity (CIC deposit -> FFT Poisson solve -> gradient -> CIC gather),
// the P3M short-range correction, AND collision - all as compute shaders over storage
// buffers. Positions/velocities never leave GPU memory: webgpuRenderer.ts's sim-buffer
// render path draws directly from this pipeline's position buffer, so the only recurring
// GPU->CPU traffic is an 8-byte max-acceleration/speed readback (drives the adaptive
// substep count and neighbor-grid cell sizing, both one frame stale by design - WebGPU
// readbacks are async, which fits that staleness contract exactly).
//
// Differences from the native OpenGL version, all forced by WebGPU/WGSL rather than
// chosen:
//  - The FFT row kernel uses a fixed 256-thread workgroup that loops over its row's
//    butterflies (WebGPU's default maxComputeInvocationsPerWorkgroup is 256; the native
//    version used one thread per butterfly pair). Same math, same barrier structure.
//  - Per-frame values (dt, neighbor cell size/dims, max radius) live in one small uniform
//    buffer written once per frame; everything static (grid size, particle count, physics
//    constants) is baked into the WGSL source at pipeline-build time, mirroring the native
//    version's #define header.
//  - Buffer zeroing uses GPUCommandEncoder.clearBuffer; the velocity snapshot uses
//    copyBufferToBuffer - no dedicated shaders needed.
//  - The neighbor grid covers 1.5x the window (native: 3x) and MAX_PER_CELL is 32
//    (native: 64), keeping the worst-case cell-items store inside WebGPU's default
//    128MB storage-binding limit at 1M particles / fine meshes. Escaped particles clamp
//    into edge cells; overflowing cells drop the excess for one substep - graceful
//    degradation, same policy as the native version.
//
// Physics parity notes (same as the native pipeline's own header):
//  - Collision is Jacobi-style (frozen begin-of-pass velocities, self-half application,
//    symmetric under-relaxation from the PREVIOUS substep's contact counts) rather than
//    collide.ts's sequential impulses - a GPU can't see "earlier pairs' updated
//    velocities" without serializing. Both of the native version's hard-won fixes are
//    preserved: frozen velSnap reads on BOTH sides of a pair (momentum symmetry), and
//    omega = 1/max(prevContacts_i, prevContacts_j) under-relaxation (energy boundedness
//    in collapsed cores - without it the parallel sum of k simultaneous full-strength
//    impulses pumps unbounded energy, measured directly in the native app).
//  - The mesh solve matches pmGravity.ts end-to-end (same CIC weights, same radix-2 FFT
//    math, same central-difference gradient), INCLUDING the isolated (free-space)
//    boundary treatment: FFT grids are zero-padded to 2x per axis and multiplied by the
//    same free-space kernel spectrum pmGravity.ts's buildIsolatedKernelTable computes
//    (uploaded pre-transposed) - no periodic image forces (see pmGravity.ts's header for
//    the anisotropic-collapse artifact those caused). The CPU-calibrated P3M table stays
//    valid on GPU. Padded rows need PAD_N*8 bytes of workgroup memory in the FFT kernel -
//    16KB (WebGPU's default limit) covers grids up to 1024; the worker requests a higher
//    limit from the adapter so 2048 works where hardware allows (see autoGridSize).
import { createPMGrid, buildIsolatedKernelTable } from './pmGravity.ts';

function roundUpPow2(v) {
    let p = 1;
    while (p < v) p <<= 1;
    return p;
}

/**
 * Grid resolution matched to particle count (same auto-scaling rule as the native app),
 * capped so the FFT kernel's shared row fits the device's workgroup-storage limit: the
 * isolated-boundary solve runs padded rows of 2*gridN vec2f's, i.e. gridN*16 bytes.
 */
export function autoGridSize(particleCount, maxWorkgroupStorageBytes = 16384) {
    const n = roundUpPow2(Math.ceil(Math.sqrt(particleCount * 2)));
    const deviceCap = roundUpPow2(Math.floor(maxWorkgroupStorageBytes / 16) + 1) >> 1; // largest pow2 with gridN*16 <= limit
    return Math.max(256, Math.min(n, Math.min(2048, deviceCap)));
}

/**
 * Builds the full compute pipeline over an already-spawned CPU-side particle system.
 * `table` is a calibrated PMShortRangeTable from pmGravity.ts. Throws on pipeline
 * compilation failure - the caller falls back to the CPU PM solver.
 */
export async function createWebGPUSim(device, system, opts) {
    const {
        gridN, worldW, worldH,
        pmG, pairSofteningFactor,
        treeG, treeSofteningFactor,
        restitution, surfaceGap,
        table, densityBlurThreshold,
        substepSafetyFactor, maxSubsteps,
    } = opts;

    const count = system.count;
    const cellW = worldW / gridN;
    const cellH = worldH / gridN;
    const rCut = table.rCut;
    // Mesh buffers are sized for the ZERO-PADDED (2x per axis) isolated-boundary solve -
    // see pmGravity.ts's header for why the padding is load-bearing.
    const padN = gridN * 2;
    const padCells = padN * padN;

    // --- Particle data, uploaded once from the CPU-side spawn ---
    const pos2 = new Float32Array(count * 2);
    const vel2 = new Float32Array(count * 2);
    const props4 = new Float32Array(count * 4); // invMass (0 = fixed), radius, mass, 0
    const colorRadius = new Float32Array(count * 4); // r, g, b, radius - render-only
    let maxParticleRadius = 0;
    let totalMass = 0;
    for (let i = 0; i < count; i++) {
        pos2[i * 2] = system.posX[i];
        pos2[i * 2 + 1] = system.posY[i];
        vel2[i * 2] = system.velX[i];
        vel2[i * 2 + 1] = system.velY[i];
        props4[i * 4] = system.fixed[i] ? 0 : 1 / system.mass[i];
        props4[i * 4 + 1] = system.radius[i];
        props4[i * 4 + 2] = system.mass[i];
        colorRadius[i * 4] = system.colorR[i] / 255;
        colorRadius[i * 4 + 1] = system.colorG[i] / 255;
        colorRadius[i * 4 + 2] = system.colorB[i] / 255;
        colorRadius[i * 4 + 3] = system.radius[i];
        maxParticleRadius = Math.max(maxParticleRadius, system.radius[i]);
        totalMass += system.mass[i];
    }

    // Fixed-point scale for the atomic CIC deposit (no float atomics in WGSL). The native
    // version hardcodes 2^24; here the scale is derated when even the absurd worst case
    // (every unit of mass in one cell - e.g. the fixed central mass on a very fine grid
    // over a small window) would overflow u32, trading unneeded precision for safety.
    const worstCellDensity = totalMass / (cellW * cellH);
    const fpScale = Math.min(2 ** 24, 4.0e9 / worstCellDensity);

    // --- Neighbor grid sizing (worst case: smallest cell = rCut), byte-budgeted ---
    const MAX_PER_CELL = 32;
    const binExtentW = worldW * 1.5;
    const binExtentH = worldH * 1.5;
    const binOriginX = -worldW * 0.25;
    const binOriginY = -worldH * 0.25;
    let cellSizeFloor = rCut;
    const ITEM_BYTE_BUDGET = 96 * 1024 * 1024;
    for (;;) {
        const cx = Math.ceil(binExtentW / cellSizeFloor) + 1;
        const cy = Math.ceil(binExtentH / cellSizeFloor) + 1;
        if (cx * cy * MAX_PER_CELL * 4 <= ITEM_BYTE_BUDGET) break;
        cellSizeFloor *= 1.5; // coarser cells: more neighbors scanned per query, never dropped correctness
    }
    const maxBinCellsX = Math.ceil(binExtentW / cellSizeFloor) + 1;
    const maxBinCellsY = Math.ceil(binExtentH / cellSizeFloor) + 1;
    const maxCells = maxBinCellsX * maxBinCellsY;

    const mkBuf = (bytes, usage, data = null) => {
        const buf = device.createBuffer({ size: Math.max(bytes, 4), usage });
        if (data) device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
        return buf;
    };
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

    const posBuf = mkBuf(count * 8, S | CD | CS, pos2);
    const velBuf = mkBuf(count * 8, S | CD | CS, vel2);
    const accBuf = mkBuf(count * 8, S | CD);
    const propsBuf = mkBuf(count * 16, S | CD, props4);
    const velSnapBuf = mkBuf(count * 8, S | CD);
    const velDeltaBuf = mkBuf(count * 16, S);
    const densityOutBuf = mkBuf(count * 4, S);
    const contactBufA = mkBuf(count * 4, S | CD);
    const contactBufB = mkBuf(count * 4, S | CD);
    // Density color mode's local-crowding signal (see the collide shader's candidateCount
    // and applySrc below) - deliberately a SEPARATE tally from contactBufA/B: those drive
    // the Jacobi under-relaxation factor (omega) and must stay based on actual resolved
    // contacts for physics correctness (see this file's header comment on why under-
    // relaxation exists at all), while collide.ts's density signal counts every neighbor
    // within the search radius regardless of whether a contact/bounce/overlap actually
    // occurred - matching that (not the contacts count) is what this buffer is for.
    const candidateCountBuf = mkBuf(count * 4, S);
    const colorRadiusBuf = mkBuf(count * 16, S | CD, colorRadius);

    const densityGridBuf = mkBuf(padCells * 4, S | CD);
    const gridReA = mkBuf(padCells * 4, S);
    const gridImA = mkBuf(padCells * 4, S);
    const gridReB = mkBuf(padCells * 4, S);
    const gridImB = mkBuf(padCells * 4, S);
    const forceBuf = mkBuf(padCells * 8, S);

    // Free-space kernel spectrum via the exact same CPU code the CPU PM path uses,
    // transposed at upload: the pipeline applies it between the two FFT axis passes,
    // i.e. while the grid sits in transposed layout.
    const greensGrid = createPMGrid(gridN, gridN, worldW, worldH);
    buildIsolatedKernelTable(greensGrid, pmG);
    const greensT = new Float32Array(padCells);
    for (let j = 0; j < padN; j++) {
        for (let i = 0; i < padN; i++) {
            greensT[i * padN + j] = greensGrid.kernelHat[j * padN + i];
        }
    }
    const greensBuf = mkBuf(padCells * 4, S | CD, greensT);

    const cellCountBuf = mkBuf(maxCells * 4, S | CD);
    const cellItemsBuf = mkBuf(maxCells * MAX_PER_CELL * 4, S | CD);
    const p3mTableBuf = mkBuf(table.tableSize * 4, S | CD, table.accPerUnitMass);
    const maxBitsBuf = mkBuf(8, S | CD | CS);
    const maxBitsStaging = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const frameUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // --- WGSL sources. Static config is baked in, mirroring the native #define header ---
    const log2N = Math.log2(gridN);
    const consts = /* wgsl */ `
const COUNT: u32 = ${count}u;
const GRID_N: u32 = ${gridN}u;
const PAD_N: u32 = ${padN}u;
const PAD_NI: i32 = ${padN};
const LOG2_PAD: u32 = ${log2N + 1}u;
const HALF_PAD: u32 = ${gridN}u;
const MAX_PER_CELL: u32 = ${MAX_PER_CELL}u;
const TABLE_SIZE: i32 = ${table.tableSize};
const PM_G: f32 = ${pmG};
const PAIR_SOFT: f32 = ${pairSofteningFactor};
const TREE_G: f32 = ${treeG};
const TREE_SOFT: f32 = ${treeSofteningFactor};
const RESTITUTION: f32 = ${restitution};
const GAP: f32 = ${surfaceGap};
const FP_SCALE: f32 = ${fpScale};
const CELL_W: f32 = ${cellW};
const CELL_H: f32 = ${cellH};
const BIN_ORIGIN: vec2f = vec2f(${binOriginX}, ${binOriginY});
const R_CUT: f32 = ${rCut};
const BLUR_THRESHOLD: f32 = ${densityBlurThreshold};
struct Frame { dt: f32, cellSize: f32, maxRadius: f32, pad0: f32, dims: vec2i, pad1: i32, pad2: i32 };
`;

    const integrateSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(2) var<storage, read> acc: array<vec2f>;
@group(0) @binding(3) var<storage, read> props: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> maxBits: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> frame: Frame;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    if (props[i].x != 0.0) { // invMass 0 = fixed particle, never moves
        vel[i] += acc[i] * frame.dt;
        pos[i] += vel[i] * frame.dt;
    }
    atomicMax(&maxBits[1], bitcast<u32>(dot(vel[i], vel[i])));
}`;

    const binSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cellItems: array<u32>;
@group(0) @binding(3) var<uniform> frame: Frame;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    let c = clamp(vec2i((pos[i] - BIN_ORIGIN) / frame.cellSize), vec2i(0), frame.dims - vec2i(1));
    let cell = u32(c.y * frame.dims.x + c.x);
    let slot = atomicAdd(&cellCount[cell], 1u);
    // Slots past MAX_PER_CELL are dropped - those neighbors are missed for one substep
    // (graceful degradation; substepping revisits next step).
    if (slot < MAX_PER_CELL) { cellItems[cell * MAX_PER_CELL + slot] = i; }
}`;

    // Jacobi collision: same swept-contact + energy-accounted-overlap math as collide.ts
    // (see that file for each formula's derivation), against FROZEN begin-of-pass
    // pos/velSnap, self-half application only, with symmetric under-relaxation - see this
    // file's header comment for why both of those properties are load-bearing.
    const collideSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read> velSnap: array<vec2f>;
@group(0) @binding(2) var<storage, read> props: array<vec4f>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> cellItems: array<u32>;
@group(0) @binding(5) var<storage, read_write> velDelta: array<vec4f>; // xy = new vel, zw = pos correction
@group(0) @binding(6) var<storage, read> prevContacts: array<u32>;
@group(0) @binding(7) var<storage, read_write> newContacts: array<u32>;
@group(0) @binding(8) var<storage, read_write> candidateCount: array<u32>;
@group(0) @binding(9) var<uniform> frame: Frame;

fn pairPE(mi: f32, mj: f32, ri: f32, rj: f32, dist: f32) -> f32 {
    let combR = ri + rj;
    let sd = dist * dist + combR * combR * TREE_SOFT;
    return -(TREE_G * mi * mj) / sqrt(sd);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    let pi = pos[i];
    let vi = velSnap[i];
    let invMi = props[i].x;
    let ri = props[i].y;
    let mi = props[i].z;
    let searchR = ri + frame.maxRadius + GAP + length(vi);
    let searchR2 = searchR * searchR;

    var dVel = vec2f(0.0);
    var myDelta = vec2f(0.0);
    var contacts = 0u;
    // Density color mode's tally (see candidateCountBuf's declaration comment) - every
    // neighbor within searchR counts, matching collide.ts's density signal exactly (that
    // broad-phase search radius query, with no further contact/bounce/overlap filtering)
    // rather than the stricter contacts count below, which under-relaxation needs kept
    // as actual resolved contacts.
    var candidates = 0u;
    let myPrevContacts = f32(max(prevContacts[i], 1u));

    let cc = vec2i((pi - BIN_ORIGIN) / frame.cellSize);
    let c0 = clamp(cc - vec2i(1), vec2i(0), frame.dims - vec2i(1));
    let c1 = clamp(cc + vec2i(1), vec2i(0), frame.dims - vec2i(1));
    for (var cy = c0.y; cy <= c1.y; cy++) {
        for (var cx = c0.x; cx <= c1.x; cx++) {
            let cell = u32(cy * frame.dims.x + cx);
            let n = min(cellCount[cell], MAX_PER_CELL);
            for (var s = 0u; s < n; s++) {
                let j = cellItems[cell * MAX_PER_CELL + s];
                if (j == i) { continue; }
                let pj = pos[j];
                let endD = pj - pi;
                if (dot(endD, endD) > searchR2) { continue; }
                candidates++;

                let vj = velSnap[j];
                let invMj = props[j].x;
                let rj = props[j].y;
                let mj = props[j].z;
                if (invMi + invMj == 0.0) { continue; }

                let relDrift = (vj - vi) * frame.dt;
                let startD = endD - relDrift;
                let touch = ri + rj + GAP;
                let cVal = dot(startD, startD) - touch * touch;
                var t = 0.0;
                var contact = startD;
                var contacted = true;
                if (cVal > 0.0) {
                    let a = dot(relDrift, relDrift);
                    if (a < 1e-9) {
                        contacted = false;
                    } else {
                        let b = 2.0 * dot(startD, relDrift);
                        let disc = b * b - 4.0 * a * cVal;
                        if (disc < 0.0) {
                            contacted = false;
                        } else {
                            t = (-b - sqrt(disc)) / (2.0 * a);
                            if (t < 0.0 || t > 1.0) { contacted = false; }
                            else { contact = startD + t * relDrift; }
                        }
                    }
                }
                if (!contacted) { continue; }

                let omega = 1.0 / max(myPrevContacts, f32(max(prevContacts[j], 1u)));

                let closing = dot(contact, vj - vi);
                var anyContact = false;
                if (closing < 0.0) {
                    // Approaching at contact: impulse bounce (collide.ts resolveImpulse), self half.
                    let cd = max(length(contact), 1e-6);
                    let nrm = contact / cd;
                    let vn = dot(vi - vj, nrm);
                    let imp = (1.0 + RESTITUTION) * vn / (invMi + invMj);
                    let dv = -imp * invMi * nrm * omega;
                    dVel += dv;
                    myDelta += (1.0 - t) * frame.dt * dv;
                    anyContact = true;
                }
                // Energy-accounted overlap separation (collide.ts resolveOverlap), self half.
                let dist = max(length(endD), 1e-6);
                if (dist < touch) {
                    let nrm = endD / dist;
                    let mu = 1.0 / (invMi + invMj);
                    let vn = dot(vi - vj, nrm);
                    let avail = 0.5 * mu * vn * vn;
                    let fullCost = pairPE(mi, mj, ri, rj, touch) - pairPE(mi, mj, ri, rj, dist);
                    var targetDist = touch;
                    if (avail < fullCost) {
                        let combR = ri + rj;
                        let softSq = combR * combR * TREE_SOFT;
                        let gm = TREE_G * mi * mj;
                        let k = 1.0 / sqrt(dist * dist + softSq) - avail / gm;
                        if (k > 0.0) { targetDist = sqrt(max(1.0 / (k * k) - softSq, 0.0)); }
                    }
                    if (targetDist > dist) {
                        let wI = invMi / (invMi + invMj);
                        myDelta -= wI * (targetDist - dist) * nrm * omega;
                        let spent = pairPE(mi, mj, ri, rj, targetDist) - pairPE(mi, mj, ri, rj, dist);
                        let disc2 = max(vn * vn - 2.0 * spent / mu, 0.0);
                        let vnNew = sign(vn) * sqrt(disc2);
                        dVel -= mu * (vn - vnNew) * invMi * nrm * omega;
                    }
                    anyContact = true;
                }
                if (anyContact) { contacts++; }
            }
        }
    }

    velDelta[i] = vec4f(vi + dVel, myDelta);
    newContacts[i] = contacts;
    candidateCount[i] = candidates;
}`;

    const applySrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(2) var<storage, read> velDelta: array<vec4f>;
@group(0) @binding(3) var<storage, read> candidateCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> densityOut: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    vel[i] = velDelta[i].xy;
    pos[i] += velDelta[i].zw;
    // sqrt so the density color ramp's early stops get most of the visible range and only
    // truly extreme crowding reaches the final stop - see collide.ts's identical CPU-path
    // formula for why a plain linear ratio reads as one flat block of solid color instead.
    // candidateCount (not the collide pass's contacts tally - see that buffer's own
    // declaration comment) matches collide.ts's density signal: every neighbor within the
    // search radius, not just the ones that actually bounced or overlapped.
    densityOut[i] = sqrt(min(f32(candidateCount[i]) / BLUR_THRESHOLD, 1.0));
}`;

    const depositSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read> props: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> densityGrid: array<atomic<u32>>;
fn wrapIdx(v: i32) -> i32 { return ((v % PAD_NI) + PAD_NI) % PAD_NI; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    let g = pos[i] / vec2f(CELL_W, CELL_H);
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    let x0 = wrapIdx(i0.x); let y0 = wrapIdx(i0.y);
    let x1 = wrapIdx(i0.x + 1); let y1 = wrapIdx(i0.y + 1);
    let m = props[i].z / (CELL_W * CELL_H);
    atomicAdd(&densityGrid[y0 * PAD_NI + x0], u32((1.0 - f.x) * (1.0 - f.y) * m * FP_SCALE));
    atomicAdd(&densityGrid[y0 * PAD_NI + x1], u32(f.x * (1.0 - f.y) * m * FP_SCALE));
    atomicAdd(&densityGrid[y1 * PAD_NI + x0], u32((1.0 - f.x) * f.y * m * FP_SCALE));
    atomicAdd(&densityGrid[y1 * PAD_NI + x1], u32(f.x * f.y * m * FP_SCALE));
}`;

    const dens2ComplexSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> densityGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> re: array<f32>;
@group(0) @binding(2) var<storage, read_write> im: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    // 2D dispatch flattened to a linear cell index: padded grids reach 4096^2/256 = 65536
    // workgroups, one past the 65535 per-dimension dispatch limit.
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= PAD_N * PAD_N) { return; }
    re[i] = f32(densityGrid[i]) / FP_SCALE;
    im[i] = 0.0;
}`;

    // One workgroup per row, whole row in shared memory, radix-2 butterflies with barriers
    // between stages - the same math as fft.ts's fft1d. Unlike the native version (one
    // thread per butterfly), a fixed 256-thread workgroup loops over the row's butterflies:
    // WebGPU's default invocations-per-workgroup limit is 256, and grid rows go up to 2048.
    // Butterflies within one stage touch disjoint index pairs, so intra-stage order doesn't
    // matter; the workgroupBarrier between stages is what correctness actually needs.
    const fftRowSrc = (dirSign, outScale) => consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> im: array<f32>;
var<workgroup> s: array<vec2f, ${padN}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let row = wid.x;
    let t = lid.x;
    let base = row * PAD_N;
    for (var e = t; e < PAD_N; e += 256u) {
        let r = reverseBits(e) >> (32u - LOG2_PAD);
        s[r] = vec2f(re[base + e], im[base + e]);
    }
    workgroupBarrier();
    for (var len = 2u; len <= PAD_N; len = len << 1u) {
        let half_ = len >> 1u;
        for (var q = t; q < HALF_PAD; q += 256u) {
            let blk = q / half_;
            let k = q % half_;
            let i0 = blk * len + k;
            let i1 = i0 + half_;
            let ang = ${dirSign} * 6.28318530718 * f32(k) / f32(len);
            let w = vec2f(cos(ang), sin(ang));
            let a = s[i0];
            let b = s[i1];
            let bw = vec2f(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);
            s[i0] = a + bw;
            s[i1] = a - bw;
        }
        workgroupBarrier();
    }
    for (var e = t; e < PAD_N; e += 256u) {
        re[base + e] = s[e].x * ${outScale};
        im[base + e] = s[e].y * ${outScale};
    }
}`;

    const transposeSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
var<workgroup> tile: array<array<f32, 17>, 16>; // +1 pad kills shared-memory bank conflicts
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let g = vec2i(wid.xy) * 16 + vec2i(lid.xy);
    if (g.x < PAD_NI && g.y < PAD_NI) {
        tile[lid.y][lid.x] = src[g.y * PAD_NI + g.x];
    }
    workgroupBarrier();
    let gT = vec2i(wid.yx) * 16 + vec2i(lid.xy);
    if (gT.x < PAD_NI && gT.y < PAD_NI) {
        dst[gT.y * PAD_NI + gT.x] = tile[lid.x][lid.y];
    }
}`;

    const greensSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> im: array<f32>;
@group(0) @binding(2) var<storage, read> greens: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    // 2D dispatch flattened to a linear cell index: padded grids reach 4096^2/256 = 65536
    // workgroups, one past the 65535 per-dimension dispatch limit.
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= PAD_N * PAD_N) { return; }
    re[i] *= greens[i];
    im[i] *= greens[i];
}`;

    const gradientSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> force: array<vec2f>;
fn wrapIdx(v: i32) -> i32 { return ((v % PAD_NI) + PAD_NI) % PAD_NI; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (idx >= PAD_N * PAD_N) { return; }
    let x = i32(idx) % PAD_NI;
    let y = i32(idx) / PAD_NI;
    let pxm = phi[y * PAD_NI + wrapIdx(x - 1)];
    let pxp = phi[y * PAD_NI + wrapIdx(x + 1)];
    let pym = phi[wrapIdx(y - 1) * PAD_NI + x];
    let pyp = phi[wrapIdx(y + 1) * PAD_NI + x];
    force[idx] = vec2f(-(pxp - pxm) / (2.0 * CELL_W), -(pyp - pym) / (2.0 * CELL_H));
}`;

    const gatherSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read> force: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec2f>;
fn wrapIdx(v: i32) -> i32 { return ((v % PAD_NI) + PAD_NI) % PAD_NI; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    let g = pos[i] / vec2f(CELL_W, CELL_H);
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    let x0 = wrapIdx(i0.x); let y0 = wrapIdx(i0.y);
    let x1 = wrapIdx(i0.x + 1); let y1 = wrapIdx(i0.y + 1);
    acc[i] = (1.0 - f.x) * (1.0 - f.y) * force[y0 * PAD_NI + x0]
           + f.x * (1.0 - f.y) * force[y0 * PAD_NI + x1]
           + (1.0 - f.x) * f.y * force[y1 * PAD_NI + x0]
           + f.x * f.y * force[y1 * PAD_NI + x1];
}`;

    // Same correction formula + linear table interpolation as pmGravity.ts's
    // applyP3MCorrection (write-only-to-self, so no atomics on acc).
    const p3mSrc = consts + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec2f>;
@group(0) @binding(1) var<storage, read> props: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec2f>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> cellItems: array<u32>;
@group(0) @binding(5) var<storage, read> meshTable: array<f32>;
@group(0) @binding(6) var<storage, read_write> maxBits: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> frame: Frame;
fn tableLookup(r: f32) -> f32 {
    let p = r / R_CUT * f32(TABLE_SIZE) - 0.5;
    if (p <= 0.0) { return meshTable[0]; }
    let k = i32(p);
    if (k >= TABLE_SIZE - 1) { return meshTable[TABLE_SIZE - 1]; }
    let fr = p - f32(k);
    return meshTable[k] * (1.0 - fr) + meshTable[k + 1] * fr;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= COUNT) { return; }
    let pi = pos[i];
    let ri = props[i].y;
    var sum = vec2f(0.0);
    let cc = vec2i((pi - BIN_ORIGIN) / frame.cellSize);
    let c0 = clamp(cc - vec2i(1), vec2i(0), frame.dims - vec2i(1));
    let c1 = clamp(cc + vec2i(1), vec2i(0), frame.dims - vec2i(1));
    for (var cy = c0.y; cy <= c1.y; cy++) {
        for (var cx = c0.x; cx <= c1.x; cx++) {
            let cell = u32(cy * frame.dims.x + cx);
            let n = min(cellCount[cell], MAX_PER_CELL);
            for (var s = 0u; s < n; s++) {
                let j = cellItems[cell * MAX_PER_CELL + s];
                if (j == i) { continue; }
                let d = pos[j] - pi;
                let r = length(d);
                if (r <= 0.0 || r >= R_CUT) { continue; }
                let combR = ri + props[j].y;
                let softenedR = sqrt(r * r + combR * combR * PAIR_SOFT);
                let mj = props[j].z;
                let corr = 2.0 * PM_G * mj / softenedR - mj * tableLookup(r);
                sum += corr * d / r;
            }
        }
    }
    let a = acc[i] + sum;
    acc[i] = a;
    atomicMax(&maxBits[0], bitcast<u32>(dot(a, a)));
}`;

    const makePipeline = (src, label) => device.createComputePipelineAsync({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label, code: src }), entryPoint: 'main' },
    });

    const [
        pipeIntegrate, pipeBin, pipeCollide, pipeApply, pipeDeposit,
        pipeDens2Complex, pipeFftFwd, pipeFftInv, pipeTranspose,
        pipeGreens, pipeGradient, pipeGather, pipeP3m,
    ] = await Promise.all([
        makePipeline(integrateSrc, 'integrate'),
        makePipeline(binSrc, 'bin'),
        makePipeline(collideSrc, 'collide'),
        makePipeline(applySrc, 'apply'),
        makePipeline(depositSrc, 'deposit'),
        makePipeline(dens2ComplexSrc, 'dens2complex'),
        makePipeline(fftRowSrc('-1.0', '1.0'), 'fft_fwd'),
        makePipeline(fftRowSrc('1.0', `${1 / padN}`), 'fft_inv'),
        makePipeline(transposeSrc, 'transpose'),
        makePipeline(greensSrc, 'greens'),
        makePipeline(gradientSrc, 'gradient'),
        makePipeline(gatherSrc, 'gather'),
        makePipeline(p3mSrc, 'p3m'),
    ]);

    const bg = (pipeline, entries) => device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });

    const bgIntegrate = bg(pipeIntegrate, [posBuf, velBuf, accBuf, propsBuf, maxBitsBuf, frameUniform]);
    const bgBin = bg(pipeBin, [posBuf, cellCountBuf, cellItemsBuf, frameUniform]);
    // Two collide bind groups: contact counts ping-pong between substeps (the pass reads
    // LAST substep's counts - frozen, so both sides of a pair derive the identical
    // under-relaxation factor - and writes this substep's). candidateCountBuf doesn't
    // ping-pong - it's write-only-then-read-once-per-substep (apply), no cross-substep
    // dependency the way the omega under-relaxation factor has.
    const bgCollideAB = bg(pipeCollide, [posBuf, velSnapBuf, propsBuf, cellCountBuf, cellItemsBuf, velDeltaBuf, contactBufA, contactBufB, candidateCountBuf, frameUniform]);
    const bgCollideBA = bg(pipeCollide, [posBuf, velSnapBuf, propsBuf, cellCountBuf, cellItemsBuf, velDeltaBuf, contactBufB, contactBufA, candidateCountBuf, frameUniform]);
    // No longer ping-ponged (candidateCountBuf replaced the old contactBufA/B read here) -
    // one bind group is enough now that apply doesn't depend on which contact buffer is
    // "previous" this substep.
    const bgApply = bg(pipeApply, [posBuf, velBuf, velDeltaBuf, candidateCountBuf, densityOutBuf]);
    const bgDeposit = bg(pipeDeposit, [posBuf, propsBuf, densityGridBuf]);
    const bgDens2Complex = bg(pipeDens2Complex, [densityGridBuf, gridReA, gridImA]);
    const bgFftA_fwd = bg(pipeFftFwd, [gridReA, gridImA]);
    const bgFftB_fwd = bg(pipeFftFwd, [gridReB, gridImB]);
    const bgFftB_inv = bg(pipeFftInv, [gridReB, gridImB]);
    const bgFftA_inv = bg(pipeFftInv, [gridReA, gridImA]);
    const bgTransReAB = bg(pipeTranspose, [gridReA, gridReB]);
    const bgTransImAB = bg(pipeTranspose, [gridImA, gridImB]);
    const bgTransReBA = bg(pipeTranspose, [gridReB, gridReA]);
    const bgTransImBA = bg(pipeTranspose, [gridImB, gridImA]);
    const bgGreens = bg(pipeGreens, [gridReB, gridImB, greensBuf]);
    const bgGradient = bg(pipeGradient, [gridReA, forceBuf]);
    const bgGather = bg(pipeGather, [posBuf, forceBuf, accBuf]);
    const bgP3m = bg(pipeP3m, [posBuf, propsBuf, accBuf, cellCountBuf, cellItemsBuf, p3mTableBuf, maxBitsBuf, frameUniform]);

    const pGroups = Math.ceil(count / 256);
    // Cell passes split over a 2D dispatch - see the shader-side index flattening.
    const cGroupsTotal = Math.ceil(padCells / 256);
    const cGroupsX = Math.min(cGroupsTotal, 32768);
    const cGroupsY = Math.ceil(cGroupsTotal / cGroupsX);
    const tGroups = Math.ceil(padN / 16);

    const sim = {
        device,
        count,
        gridN,
        posBuf,
        colorRadiusBuf,
        densityOutBuf,
        maxParticleRadius,
        representativeRadius: system.radius[0],
        lastMaxAccel: 0,
        lastMaxSpeed: 0,
        contactPing: false,
        readbackInFlight: false,
        positionStaging: null,
        destroyed: false,

        /**
         * Adaptive substep count from last frame's peak acceleration - same formula as
         * the native main loop (and the CPU path's use of last frame's acc).
         */
        computeSubsteps() {
            let substeps = 1;
            if (this.lastMaxAccel > 0 && this.representativeRadius > 0) {
                substeps = Math.ceil(Math.sqrt(this.lastMaxAccel / (substepSafetyFactor * this.representativeRadius)));
            }
            return Math.max(1, Math.min(substeps, maxSubsteps));
        },

        /**
         * Writes the per-frame uniform (shared by every substep this frame) and returns
         * the substep dt. Neighbor cell size covers both collision's speed-padded reach
         * and P3M's rCut, from last frame's max speed - one frame stale by design.
         */
        beginFrame(substeps) {
            const dt = 1 / substeps;
            const reach = 2 * maxParticleRadius + surfaceGap + this.lastMaxSpeed * 1.5 + 1;
            const cellSize = Math.max(cellSizeFloor, reach);
            const dimsX = Math.min(Math.ceil(binExtentW / cellSize) + 1, maxBinCellsX);
            const dimsY = Math.min(Math.ceil(binExtentH / cellSize) + 1, maxBinCellsY);
            const data = new ArrayBuffer(32);
            const f = new Float32Array(data);
            const iv = new Int32Array(data);
            f[0] = dt;
            f[1] = cellSize;
            f[2] = maxParticleRadius;
            iv[4] = dimsX;
            iv[5] = dimsY;
            device.queue.writeBuffer(frameUniform, 0, data);
            return dt;
        },

        /** One kick-drift-collide-gravity cycle - mirrors the native substep() dispatch for dispatch. */
        encodeSubstep(encoder) {
            const dispatch = (pipeline, group, x, y = 1) => {
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.dispatchWorkgroups(x, y, 1);
                pass.end();
            };

            // 1. kick + drift
            dispatch(pipeIntegrate, bgIntegrate, pGroups);
            // 2. freeze velocities for the Jacobi collision pass
            encoder.copyBufferToBuffer(velBuf, 0, velSnapBuf, 0, count * 8);
            // 3. clear per-substep grids
            encoder.clearBuffer(cellCountBuf);
            encoder.clearBuffer(densityGridBuf);
            // 4. bin particles into the neighbor grid
            dispatch(pipeBin, bgBin, pGroups);
            // 5-6. collide (Jacobi, contact ping-pong across substeps) + apply
            dispatch(pipeCollide, this.contactPing ? bgCollideBA : bgCollideAB, pGroups);
            dispatch(pipeApply, bgApply, pGroups);
            this.contactPing = !this.contactPing;
            // 7. CIC deposit from corrected positions
            dispatch(pipeDeposit, bgDeposit, pGroups);
            // 8. fixed-point density -> complex field
            dispatch(pipeDens2Complex, bgDens2Complex, cGroupsX, cGroupsY);
            // 9-15. FFT Poisson solve: rowFFT -> transpose -> rowFFT -> greens (in
            // transposed layout, table pre-transposed at init) -> inverse rowFFT ->
            // transpose -> inverse rowFFT, 1/N-per-axis scaling folded into the inverses.
            dispatch(pipeFftFwd, bgFftA_fwd, padN);
            dispatch(pipeTranspose, bgTransReAB, tGroups, tGroups);
            dispatch(pipeTranspose, bgTransImAB, tGroups, tGroups);
            dispatch(pipeFftFwd, bgFftB_fwd, padN);
            dispatch(pipeGreens, bgGreens, cGroupsX, cGroupsY);
            dispatch(pipeFftInv, bgFftB_inv, padN);
            dispatch(pipeTranspose, bgTransReBA, tGroups, tGroups);
            dispatch(pipeTranspose, bgTransImBA, tGroups, tGroups);
            dispatch(pipeFftInv, bgFftA_inv, padN);
            // 16. force field = -grad(phi)
            dispatch(pipeGradient, bgGradient, cGroupsX, cGroupsY);
            // 17. CIC gather -> acc
            dispatch(pipeGather, bgGather, pGroups);
            // 18. P3M short-range correction onto acc (+ records max |acc|^2)
            dispatch(pipeP3m, bgP3m, pGroups);
        },

        /** Copy this frame's max-accel/speed accumulator out and reset it - encode last. */
        encodeReadback(encoder) {
            if (!this.readbackInFlight) {
                encoder.copyBufferToBuffer(maxBitsBuf, 0, maxBitsStaging, 0, 8);
            }
            encoder.clearBuffer(maxBitsBuf);
        },

        /**
         * Kick off the async max-bits readback after submit. WebGPU readbacks can't be
         * synchronous; the value lands a frame or two later, which matches the "last
         * frame's acceleration drives this frame's substeps" contract the CPU path
         * already has.
         */
        pollReadback() {
            if (this.readbackInFlight || this.destroyed) return;
            this.readbackInFlight = true;
            maxBitsStaging.mapAsync(GPUMapMode.READ).then(() => {
                if (this.destroyed) return;
                const bits = new Uint32Array(maxBitsStaging.getMappedRange().slice(0));
                maxBitsStaging.unmap();
                const f = new Float32Array(bits.buffer);
                this.lastMaxAccel = Math.sqrt(Math.max(f[0], 0));
                this.lastMaxSpeed = Math.sqrt(Math.max(f[1], 0));
                this.readbackInFlight = false;
            }).catch(() => { this.readbackInFlight = false; });
        },

        /**
         * Occasional diagnostics readback (debug-panel center of mass, ~once per second) -
         * an 8MB round-trip at 1M particles, deliberately not on the per-frame path.
         */
        async readPositions() {
            if (!this.positionStaging) {
                this.positionStaging = device.createBuffer({
                    size: count * 8,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                });
            }
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(posBuf, 0, this.positionStaging, 0, count * 8);
            device.queue.submit([encoder.finish()]);
            await this.positionStaging.mapAsync(GPUMapMode.READ);
            const out = new Float32Array(this.positionStaging.getMappedRange().slice(0));
            this.positionStaging.unmap();
            return out;
        },

        destroy() {
            this.destroyed = true;
            for (const b of [posBuf, velBuf, accBuf, propsBuf, velSnapBuf, velDeltaBuf,
                             densityOutBuf, contactBufA, contactBufB, candidateCountBuf, colorRadiusBuf,
                             densityGridBuf, gridReA, gridImA, gridReB, gridImB, forceBuf,
                             greensBuf, cellCountBuf, cellItemsBuf, p3mTableBuf, maxBitsBuf,
                             maxBitsStaging, frameUniform]) {
                b.destroy();
            }
            if (this.positionStaging) this.positionStaging.destroy();
        },
    };

    // Prime acc on GPU: one full substep at dt=0 - integration moves nothing, but the
    // gravity chain fills accBuf so the first real substep's kick has a force (same
    // priming step as the native init).
    sim.beginFrame(Infinity); // dt = 0
    const primeEncoder = device.createCommandEncoder();
    sim.encodeSubstep(primeEncoder);
    sim.encodeReadback(primeEncoder);
    device.queue.submit([primeEncoder.finish()]);
    sim.pollReadback();

    return sim;
}
