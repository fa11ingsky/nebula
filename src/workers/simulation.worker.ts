// Owns the entire hot path - physics (kick/drift/merge/gravity) AND rendering - off the
// main thread, so neither one can ever cause the UI (settings panel, resize handling,
// Vue reactivity) to jank, and so the two together get genuine OS-thread parallelism
// against whatever else the main thread is doing.
//
// This worker receives an OffscreenCanvas (transferred once from Particles.vue via
// canvas.transferControlToOffscreen()) and draws directly to it every tick - after the
// transfer, the main thread can no longer touch that canvas's pixels at all, which is
// exactly what makes this "real" off-main-thread rendering rather than just off-main-
// thread computation with the drawing still happening back on the main thread.
//
// No SharedArrayBuffer: that would need Cross-Origin-Opener-Policy/Cross-Origin-
// Embedder-Policy response headers to enable cross-origin isolation, which GitHub Pages
// (this app's hosting) has no way to set. Instead, state lives entirely inside this
// worker - the main thread never touches the particle data at all, only forwards
// user-driven settings changes in and receives small, infrequent debug-panel stats back
// out. That sidesteps the missing-headers problem entirely: there's no per-frame data to
// transfer either direction, since the worker both computes AND draws every frame itself.
//
// Every function imported from lib/sim/ below is exactly the same code Particles.vue used
// to call directly on the main thread - none of it imports p5 or touches window/document,
// so it runs completely unmodified in a worker. The one piece that previously came from a
// real p5 instance (random/drawingContext/frameCount/noStroke/fill/ellipse) is replaced
// by the small `p5Shim` object below, which implements just that handful of methods
// against a raw CanvasRenderingContext2D - see particleRender.ts/spawn.ts for the exact
// calls it needs to satisfy.
//
// Two rendering backends: the default Canvas2D path (full textured look, via p5Shim +
// particleRender.ts) and an optional WebGPU path (see webgpuRenderer.ts - flat-colored
// circles only, but drawn via one instanced GPU draw call for the whole swarm instead of
// per-particle Canvas2D calls). A <canvas> can only ever be given ONE context type for
// its lifetime, so switching backends means the main thread hands over a brand new
// OffscreenCanvas (see the 'switchRenderer' case) rather than this worker reconfiguring
// its existing one.
import constants from '../lib/constants.ts';
import * as simulation from '../lib/sim/simulation.ts';
import { createWebGPURenderer, resizeWebGPURenderer, setWebGPUBackground, renderWebGPUFrame, renderWebGPUSimFrame, attachSimBuffers, destroyWebGPURenderer } from '../lib/sim/webgpuRenderer.ts';
import { createPMGrid, createPMScratch, calibratePMShortRangeTable } from '../lib/sim/pmGravity.ts';
import { createWebGPUSim, autoGridSize } from '../lib/sim/webgpuSim.ts';

// Fire-and-forget: gravity.ts's computeGravity checks readiness itself and transparently
// runs the plain JS Barnes-Hut path until this resolves, so nothing here needs to await it
// - worst case, the first frame or two of a session use JS gravity instead of WASM.
simulation.initGravityWasm();

let canvas = null;
let ctx = null;
let backgroundBitmap = null;
let state = null;
let explosions = [];
let worldCenter = { x: 0, y: 0 };

let stopped = false;
let centralMassEnabled = false;
let mergingEnabled = false;
let texturesEnabled = false;
let crosshairVisible = false;
let explosionsEnabled = false;
let debugPanelVisible = false;
// Color every particle by local crowding (colors.ts's densityRamp) instead of its own
// color - see simulation.ts's displayAll (CPU paths, driven by collide.ts's per-frame
// candidate count) and webgpuRenderer.ts's renderWebGPUSimFrame (GPU path, driven by
// webgpuSim.ts's contact-derived densityOutBuf).
let densityColors = false;

// Which gravity solver runs each frame - 'tree' (Barnes-Hut, the original path), 'pm'
// (CPU Particle-Mesh, pmGravity.ts) or 'gpu' (the full WebGPU physics pipeline,
// webgpuSim.ts). See constants.ts's GRAVITY_SOLVER comment for what each one is.
let gravityMode = constants.GRAVITY_SOLVER;
// PM-mode state, rebuilt at every spawn: { grid, table, scratch, G, pairSofteningFactor }
// handed to stepSimulation. The calibration table is cached across restarts (it only
// depends on cell size / G / cutoff, not on the particles).
let pmContext = null;
const calibrationCache = new Map();
const PM_CPU_GRID = 256;
// GPU-mode state. The device is a lazily-created singleton shared between the physics
// pipeline and the WebGPU renderer - buffers are never shareable across devices, and the
// whole point is rendering straight from the physics position buffer.
let gpuSim = null;
let gpuDevicePromise = null;
let gpuInitToken = 0; // invalidates an in-flight async init superseded by a newer reset
// Center of mass in GPU mode comes from an occasional async position readback (an 8MB
// round trip at 1M particles - deliberately ~once a second, never per frame).
const gpuStats = { comX: 0, comY: 0, lastComRead: 0, comReadInFlight: false };

function getGpuDevice() {
    if (!gpuDevicePromise) {
        gpuDevicePromise = (async () => {
            if (!navigator.gpu) return null;
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return null;
            // The isolated-boundary FFT kernel keeps one PADDED row (2*gridN vec2f's) in
            // workgroup memory - the default 16KB limit caps the mesh at grid 1024.
            // Request up to 32KB where the adapter offers it so the 2048 grid (used from
            // ~500k particles up) stays available; autoGridSize caps itself to whatever
            // the device actually granted.
            const wantedWorkgroupStorage = Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768);
            // The collide compute shader binds 9 storage buffers (pos, velSnap, props,
            // cellCount, cellItems, velDelta, prevContacts, newContacts, candidateCount -
            // see webgpuSim.ts) - one past WebGPU's default maxStorageBuffersPerShaderStage
            // of 8. Request a higher limit (adapters commonly support 10+, often 16); if
            // even the adapter's own max can't cover it, pipeline creation itself will
            // throw and initGpuSim's caller falls back to the CPU PM solver.
            const wantedStorageBuffers = Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 12);
            try {
                return await adapter.requestDevice({
                    requiredLimits: {
                        maxComputeWorkgroupStorageSize: wantedWorkgroupStorage,
                        maxStorageBuffersPerShaderStage: wantedStorageBuffers,
                    },
                });
            } catch {
                return adapter.requestDevice();
            }
        })().catch(() => null);
    }
    return gpuDevicePromise;
}

function pmEffectiveG() {
    // The GRAVITATIONAL_CONSTANT setting scales PM gravity too, on top of PM's own
    // separately-tuned base constant (see constants.ts). Baked into the Green's table and
    // calibration at spawn/init - mid-run changes take effect on Restart in PM/GPU modes.
    return constants.PM_GRAVITATIONAL_CONSTANT * constants.GRAVITATIONAL_CONSTANT;
}

function getCalibratedTable(cellW, cellH, G) {
    const rCut = constants.PM_P3M_CUTOFF_FACTOR * Math.sqrt(cellW * cellH);
    const key = `${cellW}|${cellH}|${G}|${rCut}|${constants.PM_CALIBRATION_TABLE_SIZE}`;
    let table = calibrationCache.get(key);
    if (!table) {
        table = calibratePMShortRangeTable(cellW, cellH, G, rCut, constants.PM_CALIBRATION_TABLE_SIZE);
        calibrationCache.set(key, table);
    }
    return table;
}

function buildPMContext(worldW, worldH) {
    const G = pmEffectiveG();
    const grid = createPMGrid(PM_CPU_GRID, PM_CPU_GRID, worldW, worldH);
    return {
        grid,
        table: getCalibratedTable(grid.cellWidth, grid.cellHeight, G),
        scratch: createPMScratch(),
        G,
        pairSofteningFactor: constants.PM_PAIR_SOFTENING_FACTOR,
    };
}

/**
 * Builds the WebGPU physics pipeline for the just-spawned system. Async (device request,
 * pipeline compilation) and guarded by gpuInitToken - a Restart mid-init just abandons
 * the stale pipeline. On any failure this falls back to the CPU PM solver and tells the
 * main thread so the settings radio reflects reality.
 */
async function initGpuSim() {
    const token = ++gpuInitToken;
    const spawnedState = state;
    const worldW = p5Shim.width, worldH = p5Shim.height;
    try {
        const device = await getGpuDevice();
        if (!device) throw new Error('WebGPU unavailable');
        const G = pmEffectiveG();
        const gridN = autoGridSize(spawnedState.count, device.limits.maxComputeWorkgroupStorageSize);
        const sim = await createWebGPUSim(device, spawnedState, {
            gridN,
            worldW,
            worldH,
            pmG: G,
            pairSofteningFactor: constants.PM_PAIR_SOFTENING_FACTOR,
            treeG: constants.GRAVITATIONAL_CONSTANT,
            treeSofteningFactor: constants.GRAVITY_SOFTENING_FACTOR,
            restitution: constants.COLLISION_RESTITUTION,
            surfaceGap: constants.COLLISION_SURFACE_GAP,
            table: getCalibratedTable(worldW / gridN, worldH / gridN, G),
            densityBlurThreshold: constants.DENSITY_BLUR_THRESHOLD,
            substepSafetyFactor: constants.SUBSTEP_SAFETY_FACTOR,
            maxSubsteps: constants.MAX_SUBSTEPS,
        });
        if (token !== gpuInitToken) {
            sim.destroy(); // a newer reset superseded this init while it was compiling
            return;
        }
        gpuSim = sim;
        gpuStats.comX = worldCenter.x;
        gpuStats.comY = worldCenter.y;
        if (renderBackend === 'webgpu' && gpuRenderer) {
            attachSimBuffers(gpuRenderer, gpuSim);
        }
    } catch (err) {
        console.error('WebGPU physics init failed, falling back to CPU Particle Mesh:', err);
        if (token !== gpuInitToken) return;
        gravityMode = 'pm';
        postMessage({ type: 'gravityModeFallback', mode: 'pm' });
        resetSim();
    }
}

// 'canvas2d' (default) or 'webgpu' - see the module header comment above.
let renderBackend = 'canvas2d';
let gpuRenderer = null;
// False during an async backend switch (WebGPU device/pipeline setup isn't
// instantaneous) - tick() keeps stepping physics regardless, just skips drawing until
// the new backend is actually ready, rather than drawing to a half-initialized one.
let rendererReady = true;

let tickHandle = null;
let lastTickTime = 0;
let smoothedFps = 0;

// Scroll-to-zoom camera state - the web counterpart to main.cpp's CameraState/
// scrollCallback. cameraZoom scales distances from cameraAnchorX/Y (the world point the
// camera is otherwise centered on: the live center of mass for the CPU solvers, or the
// fixed spawn-time world center for the GPU solver - see cpuTick/gpuTick); cameraPanX/Y is
// the extra screen-space offset a scroll-to-cursor zoom accumulates, updated in the
// 'wheel' message handler below. Both reset to identity on every resetSim() so a Restart
// always starts from the default framing.
const MIN_ZOOM = 0.1, MAX_ZOOM = 20;
let cameraZoom = 1;
let cameraPanX = 0, cameraPanY = 0;
// Updated once per frame by whichever tick function actually ran, so a 'wheel' event (which
// can fire many times during one scroll gesture) reads the exact anchor the last drawn
// frame used instead of paying for its own fresh center-of-mass pass.
let cameraAnchorX = 0, cameraAnchorY = 0;

/**
 * A minimal stand-in for a p5 instance, implementing only the handful of methods the
 * lib/sim/ modules actually call on it (random/drawingContext/frameCount/noStroke/fill/
 * ellipse/width/height) - see particleRender.ts and spawn.ts. Letting those modules run
 * completely unmodified here (rather than forking a worker-specific copy of the
 * rendering code) means there's exactly one implementation of "how a body is drawn" to
 * keep correct, on the main thread or in this worker. Only used for the Canvas2D backend
 * - the WebGPU backend never touches p5Shim or particleRender.ts at all.
 */
const p5Shim = {
    width: 0,
    height: 0,
    frameCount: 0,
    drawingContext: null,
    random(a, b) {
        if (b === undefined) return Math.random() * a;
        return a + Math.random() * (b - a);
    },
    noStroke() {
        // No-op: the canvas 2D calls this codebase makes (arc/fill) never call ctx.stroke(),
        // so there's nothing here to suppress in the first place.
    },
    fill(r, g, b, a) {
        // p5's default color mode is RGB 0-255 on all four channels, including alpha -
        // matches every s.fill(...) call in particleRender.ts.
        ctx.fillStyle = a === undefined ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${a / 255})`;
    },
    ellipse(x, y, d) {
        ctx.beginPath();
        ctx.arc(x, y, d / 2, 0, Math.PI * 2);
        ctx.fill();
    },
};

function resetSim() {
    explosions = [];
    // 'lite' skips per-particle Canvas2D cosmetics the GPU path can never use - see
    // particleSystem.ts's addParticle. Matters at the million-particle counts the GPU
    // solver exists for.
    const lite = gravityMode === 'gpu';
    const spawned = simulation.spawnParticles(p5Shim, centralMassEnabled, mergingEnabled, lite);
    state = spawned.state;
    worldCenter = spawned.worldCenter;

    // A fresh spawn is a fresh framing - otherwise a Restart while zoomed/panned in would
    // carry the old view over onto a swarm at a totally different visual scale.
    cameraZoom = 1;
    cameraPanX = 0;
    cameraPanY = 0;

    pmContext = null;
    if (gpuSim) {
        gpuSim.destroy();
        gpuSim = null;
    }
    gpuInitToken++; // abandon any still-compiling previous GPU init
    if (gravityMode === 'pm') {
        pmContext = buildPMContext(p5Shim.width, p5Shim.height);
    } else if (gravityMode === 'gpu') {
        initGpuSim();
    }
}

/**
 * `crosshairX/Y` is precomputed by the caller as the screen position the camera anchor
 * (COM or fixed world center - see cpuTick/gpuTick) always maps to regardless of zoom
 * (screenCenter + cameraPan) - see the wheel handler's derivation for why that's anchor-
 * and zoom-independent. Passing it in already-resolved keeps this function (and the WGSL
 * crosshair shader) from needing to know which anchor convention the caller used.
 */
function renderFrame(com, offsetX, offsetY, zoom, crosshairX, crosshairY) {
    if (renderBackend === 'webgpu') {
        if (!rendererReady || !gpuRenderer) {
            return; // mid-switch - keep stepping physics, just don't draw yet
        }
        renderWebGPUFrame(
            gpuRenderer,
            state,
            explosions,
            { offsetX, offsetY, zoom, crosshairX, crosshairY },
            crosshairVisible,
            densityColors
        );
        return;
    }

    if (backgroundBitmap) {
        ctx.drawImage(backgroundBitmap, 0, 0);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(zoom, zoom);

    simulation.displayAll(p5Shim, state, texturesEnabled, densityColors);

    for (const explosion of explosions) {
        explosion.display(p5Shim);
    }

    // Marks the center of mass - should render pinned to the middle of the screen every
    // frame, since the camera is centered on it. Arm length and stroke width are divided
    // by zoom so the marker stays a constant on-screen size instead of scaling with it -
    // ctx.scale above affects every subsequent draw call, this one included.
    if (crosshairVisible) {
        const armHalf = 8 / zoom;
        ctx.strokeStyle = 'rgb(255, 255, 255)';
        ctx.lineWidth = 1.5 / zoom;
        ctx.beginPath();
        ctx.moveTo(com.x - armHalf, com.y);
        ctx.lineTo(com.x + armHalf, com.y);
        ctx.moveTo(com.x, com.y - armHalf);
        ctx.lineTo(com.x, com.y + armHalf);
        ctx.stroke();
    }

    ctx.restore();
}

function cpuTick() {
    // One full leapfrog step (kick-drift-kick + merge detection + gravity) - see
    // simulation.ts's stepSimulation. pmContext (non-null only in 'pm' mode) swaps the
    // gravity solve from the Barnes-Hut tree to the Particle-Mesh solver.
    const stepResult = simulation.stepSimulation(state, explosions, mergingEnabled, pmContext);
    state = stepResult.state;
    const gravityTree = stepResult.gravityTree;

    if (explosionsEnabled) {
        for (const explosion of explosions) {
            explosion.update();
        }
        explosions = explosions.filter((explosion) => !explosion.isDone());
    } else {
        // Merges still happen (and still cost the same either way - collision flashes
        // are purely cosmetic, never fed back into the physics), but with the setting
        // off there's no point updating/aging/drawing whatever mergeParticles just
        // pushed in - drop it immediately instead of letting it accumulate unbounded
        // for the rest of the session.
        explosions.length = 0;
    }

    // Camera follows the center of mass, which - with total momentum held at exactly
    // zero - never moves on its own. Locking the view to it keeps the system centered
    // on screen even while individual particles get flung outward by gravity. Only the
    // particles/marker shift with the camera; the nebula backdrop is drawn unshifted,
    // as if at infinity. cameraZoom/cameraPan (see the 'wheel' handler) layer scroll-to-
    // cursor zoom on top of that same COM-centered framing.
    const com = simulation.computeCenterOfMass(state);
    cameraAnchorX = com.x;
    cameraAnchorY = com.y;
    const screenCenterX = canvas.width / 2, screenCenterY = canvas.height / 2;
    const offsetX = screenCenterX - com.x * cameraZoom + cameraPanX;
    const offsetY = screenCenterY - com.y * cameraZoom + cameraPanY;

    if (debugPanelVisible) {
        // Relative to the FIXED spawn-time world center, not the live canvas
        // dimensions the camera uses - reusing the camera's offset here was the
        // original bug this measurement had to avoid: resizing the window changes
        // canvas.width/height, which would silently redefine "center" for this
        // reading with no particle having actually moved, masquerading as huge drift.
        postMessage({
            type: 'stats',
            centerOfMass: { x: com.x - worldCenter.x, y: com.y - worldCenter.y },
            // Reuses the tree gravity already built this frame rather than a third
            // rebuild just for this readout. In PM mode there's no tree -
            // computePotentialEnergy builds its own fallback on demand.
            kineticEnergy: simulation.computeKineticEnergy(state),
            potentialEnergy: simulation.computePotentialEnergy(state, gravityTree),
            fps: smoothedFps,
            gravityBackend: pmContext
                ? `pm ${pmContext.grid.nx}x${pmContext.grid.ny} (js)`
                : simulation.getGravityBackendLabel(),
        });
    }

    renderFrame(com, offsetX, offsetY, cameraZoom, screenCenterX + cameraPanX, screenCenterY + cameraPanY);
}

/**
 * One GPU-mode frame: derive the substep count from last frame's (async-read) peak
 * acceleration, encode every substep's full compute chain plus the 8-byte max-accel/speed
 * readback into one command buffer, submit, and draw straight from the physics position
 * buffer. The CPU never touches particle state here - the camera stays pinned to the
 * spawn-time world center (the fixed central anchor of the periodic PM domain) instead of
 * chasing a per-frame center of mass, which is only refreshed ~once a second from an
 * async readback for the crosshair/debug panel.
 */
function gpuTick() {
    if (!gpuSim || !rendererReady || !gpuRenderer || renderBackend !== 'webgpu') {
        return; // still initializing (or mid renderer switch) - skip, don't stall
    }

    const substeps = gpuSim.computeSubsteps();
    gpuSim.beginFrame(substeps);
    const encoder = gpuSim.device.createCommandEncoder();
    for (let s = 0; s < substeps; s++) {
        gpuSim.encodeSubstep(encoder);
    }
    gpuSim.encodeReadback(encoder);
    gpuSim.device.queue.submit([encoder.finish()]);
    gpuSim.pollReadback();

    cameraAnchorX = worldCenter.x;
    cameraAnchorY = worldCenter.y;
    const screenCenterX = canvas.width / 2, screenCenterY = canvas.height / 2;
    const offsetX = screenCenterX - worldCenter.x * cameraZoom + cameraPanX;
    const offsetY = screenCenterY - worldCenter.y * cameraZoom + cameraPanY;
    renderWebGPUSimFrame(
        gpuRenderer,
        { offsetX, offsetY, zoom: cameraZoom, crosshairX: screenCenterX + cameraPanX, crosshairY: screenCenterY + cameraPanY },
        crosshairVisible,
        densityColors
    );

    const now = performance.now();
    if ((debugPanelVisible || crosshairVisible) && !gpuStats.comReadInFlight && now - gpuStats.lastComRead > 1000) {
        gpuStats.comReadInFlight = true;
        gpuStats.lastComRead = now;
        const forState = state;
        gpuSim.readPositions().then((xy) => {
            gpuStats.comReadInFlight = false;
            if (state !== forState) return; // restarted while the readback was in flight
            let totalMass = 0, cx = 0, cy = 0;
            for (let i = 0; i < forState.count; i++) {
                const m = forState.mass[i];
                totalMass += m;
                cx += m * xy[i * 2];
                cy += m * xy[i * 2 + 1];
            }
            if (totalMass > 0) {
                gpuStats.comX = cx / totalMass;
                gpuStats.comY = cy / totalMass;
            }
        }).catch(() => { gpuStats.comReadInFlight = false; });
    }

    if (debugPanelVisible) {
        postMessage({
            type: 'stats',
            centerOfMass: { x: gpuStats.comX - worldCenter.x, y: gpuStats.comY - worldCenter.y },
            // Energy readouts need full CPU-side state - not worth an 8MB+ readback per
            // second on top of the COM one. Reported as zero in GPU mode.
            kineticEnergy: 0,
            potentialEnergy: 0,
            fps: smoothedFps,
            gravityBackend: `pm ${gpuSim.gridN}x${gpuSim.gridN} (webgpu)`,
        });
    }
}

function tick() {
    if (!stopped && state) {
        const now = performance.now();
        if (lastTickTime > 0) {
            const instantFps = 1000 / Math.max(now - lastTickTime, 0.001);
            // Same idea as p5's own frameRate() readout: an exponential moving average
            // rather than the raw instantaneous value, which would jitter wildly frame to
            // frame and be useless for comparing "does this feel smoother" at a glance.
            smoothedFps = smoothedFps === 0 ? instantFps : smoothedFps * 0.9 + instantFps * 0.1;
        }
        lastTickTime = now;
        p5Shim.frameCount++;

        if (gravityMode === 'gpu') {
            gpuTick();
        } else {
            cpuTick();
        }
    }

    // Workers have no requestAnimationFrame (it's tied to the window's render pipeline,
    // not available on DedicatedWorkerGlobalScope), and drawing to an OffscreenCanvas
    // that's attached to a visible <canvas> via transferControlToOffscreen() presents
    // automatically on the browser's own compositing schedule without needing an explicit
    // commit() call - so a plain ~60Hz self-scheduling timer is all that's needed here,
    // and it doubles as the yield point that lets queued postMessage handlers run between
    // frames (a synchronous infinite loop would starve them - settings changes and
    // stop/resume would never be able to get through).
    tickHandle = setTimeout(tick, 16);
}

async function switchToWebGPU(newCanvas, bitmap) {
    rendererReady = false;
    // GPU-physics mode shares one device between compute and rendering (buffers can't
    // cross devices) - hand the renderer the sim's device if that mode is active.
    const sharedDevice = gravityMode === 'gpu' ? await getGpuDevice() : null;
    const renderer = await createWebGPURenderer(newCanvas, sharedDevice);
    if (!renderer) {
        // Shouldn't normally happen (Particles.vue only offers this path after its own
        // support check succeeds), but if the device/pipeline chain fails anyway, report
        // it so the main thread can fall back rather than leaving rendering stuck.
        postMessage({ type: 'rendererSwitchFailed', backend: 'webgpu' });
        rendererReady = true;
        return;
    }
    canvas = newCanvas;
    gpuRenderer = renderer;
    // Whatever device the renderer ended up on becomes THE device for any later GPU
    // physics init - the physics buffers must live on the renderer's device or its render
    // pipeline can't read them. (When gravityMode was already 'gpu', sharedDevice above
    // came from this same promise and this is a no-op.)
    if (!gpuDevicePromise) {
        gpuDevicePromise = Promise.resolve(renderer.device);
    }
    if (bitmap) {
        setWebGPUBackground(gpuRenderer, bitmap);
        bitmap.close();
    }
    // If the physics pipeline finished initializing before the renderer did (the two are
    // independently async), hook its buffers up now - initGpuSim does the same in the
    // opposite arrival order.
    if (gpuSim) {
        attachSimBuffers(gpuRenderer, gpuSim);
    }
    renderBackend = 'webgpu';
    rendererReady = true;
}

function switchToCanvas2D(newCanvas, bitmap) {
    canvas = newCanvas;
    ctx = canvas.getContext('2d');
    p5Shim.drawingContext = ctx;
    p5Shim.width = canvas.width;
    p5Shim.height = canvas.height;
    if (backgroundBitmap) {
        backgroundBitmap.close();
    }
    backgroundBitmap = bitmap;
    renderBackend = 'canvas2d';
    rendererReady = true;
}

self.onmessage = (event) => {
    const msg = event.data;

    switch (msg.type) {
        case 'init': {
            canvas = msg.canvas;
            ctx = canvas.getContext('2d');
            p5Shim.drawingContext = ctx;
            p5Shim.width = canvas.width;
            p5Shim.height = canvas.height;
            backgroundBitmap = msg.backgroundBitmap;
            centralMassEnabled = msg.centralMassEnabled;
            mergingEnabled = msg.mergingEnabled;
            texturesEnabled = msg.texturesEnabled;
            crosshairVisible = msg.crosshairVisible;
            explosionsEnabled = msg.explosionsEnabled;
            debugPanelVisible = msg.debugPanelVisible;
            densityColors = msg.densityColors;

            resetSim();

            if (tickHandle === null) {
                tick();
            }
            break;
        }

        case 'resize': {
            // Only this worker can resize the OffscreenCanvas's drawing buffer once it's
            // been transferred - the main thread can still resize the visible element's
            // CSS display size, but the buffer itself (canvas.width/height) is exclusively
            // ours now.
            if (renderBackend === 'webgpu' && gpuRenderer) {
                resizeWebGPURenderer(gpuRenderer, msg.width, msg.height);
                if (msg.backgroundBitmap) {
                    setWebGPUBackground(gpuRenderer, msg.backgroundBitmap);
                    msg.backgroundBitmap.close();
                }
            } else {
                canvas.width = msg.width;
                canvas.height = msg.height;
                if (backgroundBitmap) {
                    backgroundBitmap.close();
                }
                backgroundBitmap = msg.backgroundBitmap;
            }
            p5Shim.width = msg.width;
            p5Shim.height = msg.height;
            break;
        }

        case 'switchRenderer': {
            // Old canvas (whichever backend it belonged to) is simply dropped - Canvas2D
            // needs no explicit teardown, and an OffscreenCanvas that still has a WebGPU
            // context alive would otherwise leak GPU resources, so that case is cleaned
            // up explicitly.
            if (renderBackend === 'webgpu' && gpuRenderer) {
                destroyWebGPURenderer(gpuRenderer);
                // If that renderer owned the cached shared device, it just destroyed it -
                // a later GPU-physics init must request a fresh one, not reuse a corpse.
                if (gpuRenderer.ownsDevice) {
                    gpuDevicePromise = null;
                }
                gpuRenderer = null;
            }
            if (msg.backend === 'webgpu') {
                switchToWebGPU(msg.canvas, msg.backgroundBitmap);
            } else {
                switchToCanvas2D(msg.canvas, msg.backgroundBitmap);
            }
            break;
        }

        case 'resetSim': {
            centralMassEnabled = msg.centralMassEnabled;
            resetSim();
            break;
        }

        case 'setGravityMode': {
            gravityMode = msg.mode;
            centralMassEnabled = msg.centralMassEnabled;
            // Solver state (PM grids, calibration, GPU pipeline) is built per spawn -
            // switching solvers is always a restart, same as the particle-count setting.
            resetSim();
            break;
        }

        case 'setMergingEnabled': {
            mergingEnabled = msg.value;
            // Recolors in place rather than waiting for the next Restart - see
            // particleSystem.ts's recolorAll for why color otherwise wouldn't reflect the
            // new mode until then.
            if (state) {
                simulation.recolorAll(state, mergingEnabled);
            }
            break;
        }

        case 'setStopped': {
            stopped = msg.value;
            break;
        }

        case 'setTexturesEnabled': {
            texturesEnabled = msg.value;
            break;
        }

        case 'setCrosshairVisible': {
            crosshairVisible = msg.value;
            break;
        }

        case 'setExplosionsEnabled': {
            explosionsEnabled = msg.value;
            break;
        }

        case 'setDebugPanelVisible': {
            debugPanelVisible = msg.value;
            break;
        }

        case 'setDensityColors': {
            densityColors = msg.value;
            break;
        }

        case 'wheel': {
            // Scroll-to-cursor zoom - the web counterpart to main.cpp's scrollCallback.
            // Solve for the world point currently under the cursor at the OLD zoom, update
            // zoom, then re-solve cameraPan so that same world point still lands under the
            // cursor at the NEW zoom (the standard "zoom toward cursor" trick - see
            // main.cpp's own comment for the identical algebra). cameraAnchorX/Y is
            // whichever anchor last frame's tick actually rendered with (live COM for the
            // CPU solvers, the fixed world center for the GPU solver), read as a cached
            // value rather than recomputed here since a fast scroll gesture can fire many
            // of these between rendered frames.
            if (!canvas) break;
            const screenCenterX = canvas.width / 2, screenCenterY = canvas.height / 2;
            const worldUnderMouseX = cameraAnchorX + (msg.mouseX - screenCenterX - cameraPanX) / cameraZoom;
            const worldUnderMouseY = cameraAnchorY + (msg.mouseY - screenCenterY - cameraPanY) / cameraZoom;

            // Wheel deltaY convention: scrolling up/away from the user is negative in the
            // DOM WheelEvent spec, so negating it before the exponent makes "scroll up/
            // away zooms in" - matching main.cpp's GLFW-based scrollCallback. Dividing by
            // 100 approximates one native scroll "notch" (a 1.1x step) per ~100 units of
            // deltaY - the common step size a traditional mouse wheel reports in Chrome's
            // default DOM_DELTA_PIXEL mode; trackpads report much smaller continuous
            // deltas, which fall out of this same formula as proportionally smaller smooth
            // zoom steps rather than full notches.
            cameraZoom *= Math.pow(1.1, -msg.deltaY / 100);
            cameraZoom = Math.min(Math.max(cameraZoom, MIN_ZOOM), MAX_ZOOM);

            cameraPanX = msg.mouseX - screenCenterX - (worldUnderMouseX - cameraAnchorX) * cameraZoom;
            cameraPanY = msg.mouseY - screenCenterY - (worldUnderMouseY - cameraAnchorY) * cameraZoom;
            break;
        }

        case 'pan': {
            // Click-and-drag panning - the web counterpart to main.cpp's cursorPosCallback.
            // cameraPanX/Y is already a screen-space quantity (see the offsetX/Y formulas
            // in cpuTick/gpuTick: offset = screenCenter - anchor*zoom + pan), so the raw
            // mouse-movement delta gets added directly with no zoom conversion - unlike the
            // native app's world-space camera offset (which divides by zoom before
            // accumulating), adding screen pixels straight through is what makes the
            // content track the cursor 1:1 regardless of the current zoom level.
            cameraPanX += msg.dx;
            cameraPanY += msg.dy;
            break;
        }

        case 'setConstant': {
            // constants.ts is its own module instance inside this worker (workers don't
            // share module state with the main thread even though they import the same
            // source file) - settings changes have to be forwarded explicitly like this
            // rather than relying on a shared mutable object the way a single-thread
            // version could.
            constants[msg.key] = msg.value;
            break;
        }

        case 'setConstantAndReset': {
            constants[msg.key] = msg.value;
            centralMassEnabled = msg.centralMassEnabled;
            resetSim();
            break;
        }
    }
};
