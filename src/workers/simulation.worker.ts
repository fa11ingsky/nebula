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
import constants from '../lib/constants.ts';
import * as simulation from '../lib/sim/simulation.ts';

let canvas = null;
let ctx = null;
let backgroundBitmap = null;
let state = null;
let explosions = [];
let worldCenter = { x: 0, y: 0 };

let stopped = false;
let centralMassEnabled = false;
let mergingEnabled = true;
let texturesEnabled = false;
let crosshairVisible = false;
let explosionsEnabled = false;
let debugPanelVisible = false;

let tickHandle = null;
let lastTickTime = 0;
let smoothedFps = 0;

/**
 * A minimal stand-in for a p5 instance, implementing only the handful of methods the
 * lib/sim/ modules actually call on it (random/drawingContext/frameCount/noStroke/fill/
 * ellipse/width/height) - see particleRender.ts and spawn.ts. Letting those modules run
 * completely unmodified here (rather than forking a worker-specific copy of the
 * rendering code) means there's exactly one implementation of "how a body is drawn" to
 * keep correct, on the main thread or in this worker.
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
    const spawned = simulation.spawnParticles(p5Shim, centralMassEnabled);
    state = spawned.state;
    worldCenter = spawned.worldCenter;
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

        if (backgroundBitmap) {
            ctx.drawImage(backgroundBitmap, 0, 0);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        // One full leapfrog step (kick-drift-kick + merge detection + gravity) - see
        // simulation.ts's stepSimulation.
        const stepResult = simulation.stepSimulation(state, explosions, mergingEnabled);
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
        // as if at infinity.
        const com = simulation.computeCenterOfMass(state);
        const offsetX = canvas.width / 2 - com.x;
        const offsetY = canvas.height / 2 - com.y;

        ctx.save();
        ctx.translate(offsetX, offsetY);

        simulation.displayAll(p5Shim, state, texturesEnabled);

        for (const explosion of explosions) {
            explosion.display(p5Shim);
        }

        // Marks the center of mass - should render pinned to the middle of the screen
        // every frame, since the camera is centered on it.
        if (crosshairVisible) {
            ctx.strokeStyle = 'rgb(255, 255, 255)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(com.x - 8, com.y);
            ctx.lineTo(com.x + 8, com.y);
            ctx.moveTo(com.x, com.y - 8);
            ctx.lineTo(com.x, com.y + 8);
            ctx.stroke();
        }

        ctx.restore();

        if (debugPanelVisible) {
            postMessage({
                type: 'stats',
                // Relative to the FIXED spawn-time world center, not the live canvas
                // dimensions the camera uses - reusing the camera's offset here was the
                // original bug this measurement had to avoid: resizing the window changes
                // canvas.width/height, which would silently redefine "center" for this
                // reading with no particle having actually moved, masquerading as huge drift.
                centerOfMass: { x: com.x - worldCenter.x, y: com.y - worldCenter.y },
                // Reuses the tree gravity already built this frame rather than a third
                // rebuild just for this readout.
                kineticEnergy: simulation.computeKineticEnergy(state),
                potentialEnergy: simulation.computePotentialEnergy(state, gravityTree),
                fps: smoothedFps,
            });
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
            canvas.width = msg.width;
            canvas.height = msg.height;
            p5Shim.width = msg.width;
            p5Shim.height = msg.height;
            if (backgroundBitmap) {
                backgroundBitmap.close();
            }
            backgroundBitmap = msg.backgroundBitmap;
            break;
        }

        case 'resetSim': {
            centralMassEnabled = msg.centralMassEnabled;
            resetSim();
            break;
        }

        case 'setMergingEnabled': {
            mergingEnabled = msg.value;
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
