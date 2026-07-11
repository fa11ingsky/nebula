// The struct-of-arrays particle store and its most basic per-frame operations
// (kick/drift/reset-acceleration) and queries (center of mass).
import constants from '../constants.ts';
import { getDisplayColorForMass } from './colors.ts';
import { generateSurfaceFeatures } from './surfaceFeatures.ts';

/**
 * Struct-of-arrays particle store: every particle's position/velocity/acceleration/
 * mass/radius/color lives in a flat typed array, indexed 0..count-1, instead of each
 * particle being its own class instance with its own vector position/velocity/
 * acceleration objects. At tens of thousands of particles this matters a lot: an
 * array-of-objects layout means pointer-chasing through many small heap allocations
 * (each with its own memory address, scattered across the heap) every time a hot loop
 * (kick, drift, gravity, merge distance checks) touches a field; a typed array is one
 * single contiguous block of memory the CPU can stream through efficiently, with zero
 * per-particle object/GC overhead.
 *
 * `capacity` is fixed at creation (TOTAL_PARTICLES, +1 if a central mass is included) -
 * particles only ever get *removed* via merging within a session, never added, so there's
 * no need for dynamic resizing. `count` is the number of live particles, always occupying
 * indices [0, count) contiguously - merges compact the arrays to maintain that invariant
 * (see merge.ts) rather than leaving gaps.
 *
 * Surface texture data (craters/clouds/flares/ring) is deliberately NOT flattened into
 * typed arrays - it's a variable-shape, cosmetic-only structure that's never touched
 * unless textures are enabled, so it stays as a plain object per particle in a parallel
 * `surface` array instead.
 */
export function createParticleSystem(capacity) {
    return {
        capacity,
        count: 0,
        posX: new Float32Array(capacity),
        posY: new Float32Array(capacity),
        velX: new Float32Array(capacity),
        velY: new Float32Array(capacity),
        accX: new Float32Array(capacity),
        accY: new Float32Array(capacity),
        mass: new Float32Array(capacity),
        radius: new Float32Array(capacity),
        colorR: new Uint8Array(capacity),
        colorG: new Uint8Array(capacity),
        colorB: new Uint8Array(capacity),
        colorString: new Array(capacity).fill(''),
        // 0..1 local-crowding signal for the density color mode (see colors.ts's
        // densityRamp) - populated per frame by collide.ts's broad-phase search
        // (candidate count / constants.DENSITY_BLUR_THRESHOLD, clamped to 1). Stays at 0
        // (coolest ramp color) whenever collideParticles doesn't run this frame - merging
        // mode uses its own broad-phase search instead and never touches this field.
        density: new Float32Array(capacity),
        removed: new Uint8Array(capacity),
        // Clamped tree-insertion coordinates - see quadtree.ts's buildQuadtree.
        treeX: new Float32Array(capacity),
        treeY: new Float32Array(capacity),
        surface: new Array(capacity).fill(null),
        // 1 for a particle that never moves (currently just the optional central mass -
        // see spawn.ts) - kickAll/driftAll skip it outright, and collide.ts's
        // resolveOverlap/resolveImpulse treat it as infinitely massive (all of a
        // collision's position/velocity correction goes to the other body, none to this
        // one), rather than merely being *very* heavy and moving a little anyway.
        fixed: new Uint8Array(capacity),
    };
}

/**
 * radius defaults to the usual mass -> size mapping (sqrt(mass), so a body's rendered and
 * collision size scales with how much it weighs) but can be overridden - see spawn.ts's
 * central mass, which holds a large share of the system's total mass for gravity's sake
 * without visually/physically reading as a giant body among ordinary-sized ones.
 */
export function addParticle(system, x, y, mass, mergingEnabled = true, fixed = false, radius = null, lite = false) {
    const i = system.count;
    system.posX[i] = x;
    system.posY[i] = y;
    system.velX[i] = 0;
    system.velY[i] = 0;
    system.accX[i] = constants.GRAVITY.X;
    system.accY[i] = constants.GRAVITY.Y;
    system.mass[i] = mass;
    system.radius[i] = radius !== null ? radius : Math.sqrt(mass);
    const color = getDisplayColorForMass(mass, mergingEnabled);
    system.colorR[i] = color[0];
    system.colorG[i] = color[1];
    system.colorB[i] = color[2];
    // `lite` skips the Canvas2D-only cosmetics (a color string and a surface-feature
    // object per particle) - at the GPU solver's million-particle counts those are a
    // million heap allocations that no code path would ever read: the GPU solver requires
    // the WebGPU render backend, which draws from raw color bytes and never touches
    // either field.
    if (lite) {
        system.colorString[i] = '';
        system.surface[i] = null;
    } else {
        system.colorString[i] = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        system.surface[i] = generateSurfaceFeatures(mass);
    }
    system.density[i] = 0;
    system.fixed[i] = fixed ? 1 : 0;
    system.count++;
}

/**
 * Recolors every live particle to match a mergingEnabled toggle flipped mid-run - without
 * this, switching merging on/off wouldn't visibly change anything until the next Restart,
 * since color is otherwise only ever set at spawn time (or, in merge mode, when
 * mergeParticles actually fuses two bodies).
 */
export function recolorAll(system, mergingEnabled) {
    for (let i = 0; i < system.count; i++) {
        const color = getDisplayColorForMass(system.mass[i], mergingEnabled);
        system.colorR[i] = color[0];
        system.colorG[i] = color[1];
        system.colorB[i] = color[2];
        system.colorString[i] = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
}

export function kickAll(system, dt) {
    for (let i = 0; i < system.count; i++) {
        if (system.fixed[i]) continue;
        system.velX[i] += system.accX[i] * dt;
        system.velY[i] += system.accY[i] * dt;
    }
}

export function driftAll(system) {
    // No boundary: with nothing external ever acting on a particle, gravity and merges
    // (both exactly momentum-conserving) are the only things that can ever change total
    // system momentum - so it stays exactly constant for the whole run. A fixed particle
    // is the one deliberate exception: forcing it to never move acts as a momentum sink,
    // the same way a real collision with an immovable wall doesn't conserve just the
    // ball's own momentum - see particleSystem.ts's `fixed` field.
    for (let i = 0; i < system.count; i++) {
        if (system.fixed[i]) continue;
        system.posX[i] += system.velX[i];
        system.posY[i] += system.velY[i];
    }
}

export function resetAccelerationAll(system) {
    for (let i = 0; i < system.count; i++) {
        system.accX[i] = constants.GRAVITY.X;
        system.accY[i] = constants.GRAVITY.Y;
    }
}

/**
 * Mass-weighted center of the given particle system. If total momentum is exactly zero
 * (as spawn.ts's initializeAngularMomentum sets up), this point never moves - regardless
 * of how far individual particles wander from it.
 */
export function computeCenterOfMass(system) {
    let totalMass = 0;
    let x = 0;
    let y = 0;
    for (let i = 0; i < system.count; i++) {
        totalMass += system.mass[i];
        x += system.mass[i] * system.posX[i];
        y += system.mass[i] * system.posY[i];
    }
    return { x: x / totalMass, y: y / totalMass };
}

/** Copies every field of particle `from` onto particle `to` - used by merge.ts's compaction pass. */
export function copyParticle(system, from, to) {
    system.posX[to] = system.posX[from];
    system.posY[to] = system.posY[from];
    system.velX[to] = system.velX[from];
    system.velY[to] = system.velY[from];
    system.accX[to] = system.accX[from];
    system.accY[to] = system.accY[from];
    system.mass[to] = system.mass[from];
    system.radius[to] = system.radius[from];
    system.colorR[to] = system.colorR[from];
    system.colorG[to] = system.colorG[from];
    system.colorB[to] = system.colorB[from];
    system.colorString[to] = system.colorString[from];
    system.surface[to] = system.surface[from];
    system.density[to] = system.density[from];
    system.fixed[to] = system.fixed[from];
}
