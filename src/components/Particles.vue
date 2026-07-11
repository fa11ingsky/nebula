<template>
    <div class="simulation">
        <canvas ref="canvas" :key="canvasKey"></canvas>
        <div class="debug-panel" v-if="debugPanelVisible">
            <div>v{{ version }}</div>
            <div class="debug-title">Performance</div>
            <div>fps: {{ fps.toFixed(1) }}</div>
            <div>gravity: {{ gravityBackend }}</div>
            <div class="debug-title">Center of Mass</div>
            <div>x: {{ centerOfMass.x.toFixed(2) }}</div>
            <div>y: {{ centerOfMass.y.toFixed(2) }}</div>
            <div class="debug-title">Energy</div>
            <div>kinetic:   {{ kineticEnergy.toFixed(2) }}</div>
            <div>potential: {{ potentialEnergy.toFixed(2) }}</div>
            <div>total:     {{ (kineticEnergy + potentialEnergy).toFixed(2) }}</div>
        </div>
        <div class="settings-panel" v-if="settingsOpen">
            <label class="settings-row">
                <input type="checkbox" :checked="centralMassEnabled" @change="toggleCentralMass()" />
                Central Mass
            </label>
            <label class="settings-row" :class="{ disabled: gravitySolver === 'gpu' }">
                <input type="checkbox" :checked="mergingEnabled" @change="toggleMerging()" :disabled="gravitySolver === 'gpu'" />
                Enable Merging{{ gravitySolver === 'gpu' ? ' (not on GPU)' : '' }}
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="debugPanelVisible" />
                Show Debug Info
            </label>
            <label class="settings-row" :class="{ disabled: useWebGpu }">
                <input type="checkbox" v-model="texturesEnabled" :disabled="useWebGpu" />
                Enable Textures{{ useWebGpu ? ' (Canvas2D only)' : '' }}
            </label>
            <label class="settings-row" v-if="webgpuSupported" :class="{ disabled: gravitySolver === 'gpu' }">
                <input type="checkbox" :checked="useWebGpu" @change="toggleRenderer()" :disabled="gravitySolver === 'gpu'" />
                Use WebGPU Rendering{{ gravitySolver === 'gpu' ? ' (required by GPU solver)' : '' }}
            </label>
            <label class="settings-row" :class="{ disabled: mergingEnabled && gravitySolver !== 'gpu' }">
                <input type="checkbox" v-model="densityColors" :disabled="mergingEnabled && gravitySolver !== 'gpu'" />
                Color by Density{{ mergingEnabled && gravitySolver !== 'gpu' ? ' (non-merging only)' : '' }}
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="crosshairVisible" />
                Show Crosshair
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="explosionsEnabled" />
                Show Explosions
            </label>

            <div class="settings-group">
                <div class="settings-label">Gravity Solver</div>
                <label class="settings-row">
                    <input type="radio" name="gravitySolver" :checked="gravitySolver === 'tree'" @change="setGravitySolver('tree')" />
                    Barnes-Hut Tree (1/r&sup2;)
                </label>
                <label class="settings-row">
                    <input type="radio" name="gravitySolver" :checked="gravitySolver === 'pm'" @change="setGravitySolver('pm')" />
                    Particle Mesh (true 2D, CPU)
                </label>
                <label class="settings-row" v-if="webgpuSupported">
                    <input type="radio" name="gravitySolver" :checked="gravitySolver === 'gpu'" @change="setGravitySolver('gpu')" />
                    Particle Mesh + Collisions (WebGPU)
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Angular Momentum</div>
                <label class="settings-row" v-for="value in angularMomentumOptions" :key="'am-' + value">
                    <input type="radio" name="angularMomentum" :checked="angularMomentum === value" @change="setAngularMomentum(value)" />
                    {{ value.toLocaleString() }}
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Gravitational Constant</div>
                <label class="settings-row" v-for="value in gravitationalConstantOptions" :key="'g-' + value">
                    <input type="radio" name="gravitationalConstant" :checked="gravitationalConstant === value" @change="setGravitationalConstant(value)" />
                    {{ value }}
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Particle Count</div>
                <label class="settings-row" v-for="value in totalParticlesOptions" :key="'p-' + value">
                    <input type="radio" name="totalParticles" :checked="totalParticles === value" @change="setTotalParticles(value)" />
                    {{ value.toLocaleString() }}
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Gravity Accuracy (Barnes-Hut &theta;)</div>
                <label class="settings-row" v-for="value in barnesHutThetaOptions" :key="'bh-' + value">
                    <input type="radio" name="barnesHutTheta" :checked="barnesHutTheta === value" @change="setBarnesHutTheta(value)" />
                    {{ value }} {{ value <= 0.3 ? '(precise, slower)' : value >= 1 ? '(fast, coarser)' : '' }}
                </label>
            </div>
        </div>
        <div class="controls">
            <button id="stopButton" :class="{ active: stopped }" @click="stopped = !stopped">{{ stopped ? 'Resume' : 'Stop' }}</button>
            <button ref="restartButton" @click="resetSim()">Restart</button>
            <button :class="{ active: settingsOpen }" @click="settingsOpen = !settingsOpen">Settings</button>
        </div>
    </div>
</template>

<script>
    import p5 from 'p5';
    import constants from '../lib/constants.ts';
    import { createNebulaBackground } from '../lib/sim/nebulaBackground.ts';
    import { isWebGPUSupported } from '../lib/sim/webgpuRenderer.ts';

    export default {
        data() {
            return {
                version: constants.VERSION,
                // Bumped to force Vue to unmount/remount the <canvas> element when
                // switching rendering backends - see toggleRenderer(). Needed because a
                // canvas can only ever be given ONE context type (2d or webgpu) for its
                // entire lifetime; switching backends means handing the worker a brand
                // new, not-yet-claimed canvas rather than reusing the existing one.
                canvasKey: 0,
                // Only shown once the async support check in mounted() resolves true -
                // most browsers as of this writing don't have WebGPU (or have it behind a
                // flag), so the checkbox simply doesn't exist rather than being present
                // and non-functional.
                webgpuSupported: false,
                useWebGpu: false,
                stopped: false,
                centralMassEnabled: false,
                // With merging off (the default), colliding bodies bounce off each other
                // instead of combining - momentum- and energy-conserving elastic impulses
                // (see collide.ts) keep them apart, so the swarm jostles into tight,
                // dense clusters instead of consolidating into fewer, larger bodies.
                mergingEnabled: false,
                debugPanelVisible: false,
                settingsOpen: false,
                // Lets the surface-texture rendering (spherical shading/craters/clouds/flares/glow)
                // be switched off to fall back to plain flat circles when it's costing too
                // much at high particle counts. Off by default for that reason.
                texturesEnabled: false,
                // Color every particle by local crowding (colors.ts's densityRamp) instead
                // of its own color - see collide.ts's per-frame candidate count (CPU paths)
                // or webgpuSim.ts's contact-derived density (GPU solver). Overrides
                // texturesEnabled while active (see particleRender.ts's displayBody). On by
                // default - it's the more legible view at the higher particle counts the
                // default GPU solver (see mounted()) makes practical.
                densityColors: true,
                crosshairVisible: false,
                // Collision flashes are purely cosmetic (never fed back into the physics),
                // but at high particle counts merges happen often enough that the constant
                // flickering can be more distracting than informative - off by default.
                explosionsEnabled: false,
                // Rolling FPS reading (an exponential moving average computed inside the
                // worker, mirroring what p5's own frameRate() used to provide), shown in
                // the debug panel.
                fps: 0,
                // Mirror constants.ts's current values as the selected radio option, so the
                // settings panel opens already showing what the sim actually started with.
                angularMomentum: constants.TOTAL_ANGULAR_MOMENTUM,
                gravitationalConstant: constants.GRAVITATIONAL_CONSTANT,
                totalParticles: constants.TOTAL_PARTICLES,
                barnesHutTheta: constants.BARNES_HUT_THETA,
                // 'tree' | 'pm' | 'gpu' - see constants.ts's GRAVITY_SOLVER comment. The
                // 'gpu' choice implies (and locks) the WebGPU rendering backend: physics
                // and rendering share one GPU device and draw from the same buffers.
                gravitySolver: constants.GRAVITY_SOLVER,
                // Just the option lists for the settings panel's radio groups - defined in
                // constants.ts so adding/removing choices doesn't need a template change.
                angularMomentumOptions: constants.ANGULAR_MOMENTUM_OPTIONS,
                gravitationalConstantOptions: constants.GRAVITATIONAL_CONSTANT_OPTIONS,
                totalParticlesOptions: constants.TOTAL_PARTICLES_OPTIONS,
                barnesHutThetaOptions: constants.BARNES_HUT_THETA_OPTIONS,
                // Mirrors the worker's periodic 'stats' messages (only sent while the debug
                // panel is open) - the simulation state itself never lives on the main
                // thread at all anymore, so this is the only way these numbers get here.
                centerOfMass: { x: 0, y: 0 },
                kineticEnergy: 0,
                potentialEnergy: 0,
                gravityBackend: '...'
            };
        },

        created() {
            this.worker = null;
            // A hidden, canvas-less p5 instance kept alive purely to generate the nebula
            // background (see generateBackgroundBitmap) - the actual simulation canvas is
            // owned by the worker via OffscreenCanvas, so nothing here ever calls
            // createCanvas()/draw() on this instance.
            this.bgSketch = null;
            // Resolves once bgSketch's setup() has actually run - p5's constructor defers
            // its real internal initialization (which sets up this._elements, among other
            // things) until the window's 'load' event UNLESS document.readyState is already
            // 'complete' at construction time, so bgSketch can't safely be touched right
            // after `new p5(...)` returns. Skipping this wait used to work often enough not
            // to notice, but isn't reliable - it depends on exactly how fast the rest of the
            // page happens to load, and calling createGraphics()/remove() on a p5 instance
            // before its own setup has fired throws deep inside p5 (_pInst._elements is
            // still undefined). generateBackgroundBitmap awaits this before touching
            // bgSketch at all, on every call, not just the first.
            this.bgSketchReady = null;
            this.resizeTimer = null;
            // Click-and-drag panning state (see handleMouseDown/handleMouseMove/
            // handleMouseUp) - plain instance fields rather than reactive data, since
            // nothing in the template needs to react to a drag in progress.
            this.isDragging = false;
            this.lastMouseX = 0;
            this.lastMouseY = 0;
        },

        async mounted() {
            this.bgSketchReady = new Promise((resolve) => {
                this.bgSketch = new p5((sketch) => {
                    sketch.setup = () => {
                        sketch.noCanvas();
                        sketch.noLoop();
                        resolve();
                    };
                });
            });

            window.addEventListener('resize', this.handleResize);
            // Attached to the window (not the canvas) so a drag continues tracking even
            // once the cursor slips past the canvas's edges mid-gesture - mousedown alone
            // is canvas-scoped (see startSimulation/switchRenderer), but move/up need to
            // keep firing regardless of where the cursor ends up.
            window.addEventListener('mousemove', this.handleMouseMove);
            window.addEventListener('mouseup', this.handleMouseUp);

            // Checked and awaited BEFORE starting the sim, so the decision below is ready
            // the instant there's a worker to send it to. startSimulation() always spawns
            // in the (Barnes-Hut tree, Canvas2D) baseline first - a <canvas> only gets its
            // real backend once switchRenderer/setGravitySolver hands the worker its first
            // (and only) getContext() call, so there's no way to skip straight to a WebGPU-
            // context canvas here without duplicating that async device/pipeline setup
            // inline. Sequencing it this way (rather than firing both checks in parallel)
            // avoids a real race: setGravitySolver('gpu') below manipulates this.$refs.canvas
            // and this.worker, both of which startSimulation() is also busy setting up.
            this.webgpuSupported = await isWebGPUSupported();
            await this.startSimulation();
            if (this.webgpuSupported) {
                // Default to the full GPU pipeline (physics + rendering) when the browser
                // actually supports it - this is the only solver that stays smooth at the
                // higher particle counts, so it's the better default whenever it's available
                // rather than something the user has to discover in Settings.
                this.setGravitySolver('gpu');
            }
        },

        beforeUnmount() {
            window.removeEventListener('resize', this.handleResize);
            window.removeEventListener('mousemove', this.handleMouseMove);
            window.removeEventListener('mouseup', this.handleMouseUp);
            clearTimeout(this.resizeTimer);
            this.worker?.terminate();
            // Same readiness wait as generateBackgroundBitmap - unmounting before bgSketch's
            // setup() has fired (a fast enough navigate-away) would otherwise hit the same
            // p5-internals-not-ready crash this.bgSketchReady exists to prevent.
            this.bgSketchReady?.then(() => this.bgSketch?.remove());
        },

        methods: {
            /**
             * Renders the nebula backdrop into an offscreen p5.Graphics buffer and hands
             * back a transferable ImageBitmap - the only piece of rendering still done via
             * p5 (it needs Perlin noise, which isn't worth reimplementing), kept off the
             * hot path since it only happens at startup and on resize, not every frame.
             */
            async generateBackgroundBitmap(width, height) {
                await this.bgSketchReady;
                const graphics = createNebulaBackground(this.bgSketch, width, height);
                const bitmap = await createImageBitmap(graphics.canvas);
                // Deliberately not graphics.remove() - that calls into p5.Element.prototype's
                // internal bookkeeping (this._pInst._elements.indexOf(this)), which threw on
                // deployment (undefined _elements) for reasons that didn't reproduce from
                // reading p5's source directly. Detaching the canvas node ourselves achieves
                // the same practical goal (this offscreen buffer doesn't linger in the DOM)
                // without depending on p5 internals that turned out not to be reliable here.
                // bgSketch's own _elements array is left with a stale reference to this
                // graphics object - harmless: generateBackgroundBitmap only runs a handful of
                // times per session (startup, resize, settings changes), not per frame.
                if (graphics.canvas.parentNode) {
                    graphics.canvas.parentNode.removeChild(graphics.canvas);
                }
                return bitmap;
            },

            async startSimulation() {
                const canvasEl = this.$refs.canvas;
                const width = window.innerWidth;
                const height = window.innerHeight;
                canvasEl.width = width;
                canvasEl.height = height;
                // { passive: false } so handleWheel's preventDefault() actually stops the
                // page itself from scrolling while the cursor is over the canvas -
                // 'wheel' listeners default to passive (preventDefault a no-op) otherwise.
                canvasEl.addEventListener('wheel', this.handleWheel, { passive: false });
                canvasEl.addEventListener('mousedown', this.handleMouseDown);

                const backgroundBitmap = await this.generateBackgroundBitmap(width, height);

                // Once transferred, the main thread can never draw to (or read from) this
                // canvas again - all drawing happens in the worker from here on. This is
                // what makes rendering genuinely off-main-thread, not just physics.
                const offscreenCanvas = canvasEl.transferControlToOffscreen();

                this.worker = new Worker(new URL('../workers/simulation.worker.ts', import.meta.url), { type: 'module' });
                this.worker.onmessage = (event) => {
                    const msg = event.data;
                    if (msg.type === 'stats') {
                        this.centerOfMass = msg.centerOfMass;
                        this.kineticEnergy = msg.kineticEnergy;
                        this.potentialEnergy = msg.potentialEnergy;
                        this.fps = msg.fps;
                        this.gravityBackend = msg.gravityBackend;
                    } else if (msg.type === 'gravityModeFallback') {
                        // The worker couldn't bring up the requested solver (WebGPU device
                        // or pipeline failure) and already switched itself - reflect that
                        // in the settings radio rather than showing a solver that isn't
                        // actually running.
                        console.error(`GPU gravity solver unavailable - falling back to '${msg.mode}'.`);
                        this.gravitySolver = msg.mode;
                        constants.GRAVITY_SOLVER = msg.mode;
                    } else if (msg.type === 'rendererSwitchFailed') {
                        // Shouldn't normally happen (the settings checkbox only appears
                        // after its own support check succeeds), but if the worker's
                        // device/pipeline setup fails anyway, fall back to Canvas2D rather
                        // than leaving the canvas stuck on a half-initialized renderer.
                        console.error(`Failed to switch to ${msg.backend} rendering - falling back to Canvas2D.`);
                        this.useWebGpu = false;
                        this.switchRenderer('canvas2d');
                    }
                };

                this.worker?.postMessage({
                    type: 'init',
                    canvas: offscreenCanvas,
                    backgroundBitmap,
                    centralMassEnabled: this.centralMassEnabled,
                    mergingEnabled: this.mergingEnabled,
                    texturesEnabled: this.texturesEnabled,
                    crosshairVisible: this.crosshairVisible,
                    explosionsEnabled: this.explosionsEnabled,
                    debugPanelVisible: this.debugPanelVisible,
                    densityColors: this.densityColors,
                }, [offscreenCanvas, backgroundBitmap]);
            },

            resetSim() {
                this.worker?.postMessage({ type: 'resetSim', centralMassEnabled: this.centralMassEnabled });
            },
            /**
             * Hands the worker a brand new canvas dedicated to the given rendering
             * backend. A <canvas> can only ever be given one context type for its whole
             * lifetime, so switching backends can't just reconfigure the existing one -
             * bumping canvasKey forces Vue to unmount/remount the element first, giving a
             * fresh, not-yet-claimed canvas to transfer.
             */
            async switchRenderer(backend) {
                this.canvasKey++;
                await this.$nextTick();

                const canvasEl = this.$refs.canvas;
                const width = window.innerWidth;
                const height = window.innerHeight;
                canvasEl.width = width;
                canvasEl.height = height;
                // canvasKey's remount above means this is a brand new element - the
                // listener from the old (now-discarded) canvas doesn't carry over.
                canvasEl.addEventListener('wheel', this.handleWheel, { passive: false });
                canvasEl.addEventListener('mousedown', this.handleMouseDown);

                const backgroundBitmap = await this.generateBackgroundBitmap(width, height);
                const offscreenCanvas = canvasEl.transferControlToOffscreen();

                this.worker?.postMessage({
                    type: 'switchRenderer',
                    backend,
                    canvas: offscreenCanvas,
                    backgroundBitmap,
                }, [offscreenCanvas, backgroundBitmap]);
            },
            toggleRenderer() {
                this.useWebGpu = !this.useWebGpu;
                this.switchRenderer(this.useWebGpu ? 'webgpu' : 'canvas2d');
            },
            toggleCentralMass() {
                this.centralMassEnabled = !this.centralMassEnabled;
                this.resetSim();
            },
            toggleMerging() {
                this.mergingEnabled = !this.mergingEnabled;
                this.worker?.postMessage({ type: 'setMergingEnabled', value: this.mergingEnabled });
                this.resetSim();
            },
            setAngularMomentum(value) {
                this.angularMomentum = value;
                constants.TOTAL_ANGULAR_MOMENTUM = value;
                // Only consumed at spawn time, so it needs a restart to actually take
                // effect - bundled into one message so the worker applies the new constant
                // and restarts atomically, rather than racing two separate messages.
                this.worker?.postMessage({
                    type: 'setConstantAndReset',
                    key: 'TOTAL_ANGULAR_MOMENTUM',
                    value,
                    centralMassEnabled: this.centralMassEnabled,
                });
            },
            setGravitationalConstant(value) {
                this.gravitationalConstant = value;
                constants.GRAVITATIONAL_CONSTANT = value;
                // Read fresh every frame by the worker's gravity step, so this takes effect
                // immediately - no restart needed, unlike the other two settings here.
                this.worker?.postMessage({ type: 'setConstant', key: 'GRAVITATIONAL_CONSTANT', value });
            },
            setTotalParticles(value) {
                this.totalParticles = value;
                constants.TOTAL_PARTICLES = value;
                this.worker?.postMessage({
                    type: 'setConstantAndReset',
                    key: 'TOTAL_PARTICLES',
                    value,
                    centralMassEnabled: this.centralMassEnabled,
                });
            },
            setGravitySolver(value) {
                this.gravitySolver = value;
                constants.GRAVITY_SOLVER = value;
                // Mode first, then (if needed) the renderer switch: the worker processes
                // messages in order, so when the new WebGPU canvas arrives the worker
                // already knows to build its renderer on the physics pipeline's shared
                // device rather than requesting its own.
                this.worker?.postMessage({
                    type: 'setGravityMode',
                    mode: value,
                    centralMassEnabled: this.centralMassEnabled,
                });
                if (value === 'gpu' && !this.useWebGpu) {
                    this.useWebGpu = true;
                    this.switchRenderer('webgpu');
                }
            },
            setBarnesHutTheta(value) {
                this.barnesHutTheta = value;
                constants.BARNES_HUT_THETA = value;
                // Read fresh every frame by the worker's gravity/energy steps, so this
                // takes effect immediately - no restart needed.
                this.worker?.postMessage({ type: 'setConstant', key: 'BARNES_HUT_THETA', value });
            },

            handleResize() {
                // Resize events fire continuously while a window is being dragged; the
                // nebula background regeneration this triggers is proportional to canvas
                // area and not worth redoing dozens of times for one drag gesture, so only
                // the settled size (after a short pause) actually triggers work.
                clearTimeout(this.resizeTimer);
                this.resizeTimer = setTimeout(async () => {
                    const width = window.innerWidth;
                    const height = window.innerHeight;

                    // The canvas element's width/height attributes still control its CSS
                    // layout size on the main thread even after transferControlToOffscreen()
                    // handed the actual pixel buffer to the worker - but they no longer do
                    // anything to that buffer itself, so the worker has to resize its own
                    // OffscreenCanvas separately (see the 'resize' handler below).
                    const canvasEl = this.$refs.canvas;
                    canvasEl.width = width;
                    canvasEl.height = height;

                    const backgroundBitmap = await this.generateBackgroundBitmap(width, height);
                    this.worker?.postMessage({ type: 'resize', width, height, backgroundBitmap }, [backgroundBitmap]);
                }, 150);
            },
            /**
             * Scroll-to-zoom, toward wherever the cursor is - the web counterpart to the
             * native binary's mouse-wheel zoom. All the actual camera math (zoom level,
             * pan offset, applying it to rendering) lives in the worker, since it owns
             * all simulation/camera state; this just forwards the raw wheel delta and the
             * cursor's canvas-relative position. e.target is the canvas element itself
             * (this listener is bound directly to it), and its CSS size always matches
             * its pixel buffer size 1:1 in this app (canvas.width/height are set to
             * window.innerWidth/innerHeight with no devicePixelRatio scaling anywhere),
             * so clientX/Y minus the canvas's bounding rect is already in the same pixel
             * space the worker's own canvas.width/height use.
             */
            handleWheel(e) {
                e.preventDefault();
                const rect = e.target.getBoundingClientRect();
                this.worker?.postMessage({
                    type: 'wheel',
                    deltaY: e.deltaY,
                    mouseX: e.clientX - rect.left,
                    mouseY: e.clientY - rect.top,
                });
            },
            /**
             * Click-and-drag panning - the web counterpart to the native binary's
             * click-drag pan (see main.cpp's mouseButtonCallback/cursorPosCallback). Only
             * left-click starts a drag (button 0), matching the native app's explicit
             * GLFW_MOUSE_BUTTON_LEFT check. preventDefault stops the drag from also
             * selecting page text/triggering an image-drag ghost, a common side effect of
             * mousedown-and-move gestures over arbitrary page content.
             */
            handleMouseDown(e) {
                if (e.button !== 0) return;
                e.preventDefault();
                this.isDragging = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
            },
            /**
             * Forwards the raw screen-pixel delta since the last event - the worker's
             * cameraPanX/Y is itself already a screen-space quantity (see
             * simulation.worker.ts's 'wheel'/'pan' handlers), so unlike the native app's
             * world-space camera offset (which divides by zoom before accumulating), this
             * needs no zoom conversion here or in the worker: adding raw screen pixels
             * directly is what makes the content track the cursor 1:1 at any zoom level.
             */
            handleMouseMove(e) {
                if (!this.isDragging) return;
                const dx = e.clientX - this.lastMouseX;
                const dy = e.clientY - this.lastMouseY;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.worker?.postMessage({ type: 'pan', dx, dy });
            },
            handleMouseUp() {
                this.isDragging = false;
            },
        },

        watch: {
            stopped(value) {
                this.worker?.postMessage({ type: 'setStopped', value });
            },
            texturesEnabled(value) {
                this.worker?.postMessage({ type: 'setTexturesEnabled', value });
            },
            crosshairVisible(value) {
                this.worker?.postMessage({ type: 'setCrosshairVisible', value });
            },
            explosionsEnabled(value) {
                this.worker?.postMessage({ type: 'setExplosionsEnabled', value });
            },
            debugPanelVisible(value) {
                this.worker?.postMessage({ type: 'setDebugPanelVisible', value });
            },
            densityColors(value) {
                this.worker?.postMessage({ type: 'setDensityColors', value });
            },
        },
    };
</script>

<style scoped>
    .simulation {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
    }

    .simulation canvas {
        display: block;
        cursor: grab;
    }

    .simulation canvas:active {
        cursor: grabbing;
    }

    .debug-panel {
        position: fixed;
        bottom: 20px;
        left: 20px;
        padding: 10px 14px;
        background: rgba(10, 8, 22, 0.55);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        color: #e8e2ff;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
    }

    .debug-title {
        color: #b9a6ff;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }

    .settings-panel {
        position: fixed;
        bottom: 72px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 12px 16px;
        background: rgba(10, 8, 22, 0.55);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        color: #e8e2ff;
        font-size: 13px;
        max-height: 70vh;
        overflow-y: auto;
    }

    .settings-row {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        white-space: nowrap;
    }

    .settings-row input[type='checkbox'],
    .settings-row input[type='radio'] {
        width: 14px;
        height: 14px;
        accent-color: #b06cff;
        cursor: pointer;
    }

    .settings-row.disabled {
        opacity: 0.5;
        cursor: default;
    }

    .settings-row.disabled input {
        cursor: not-allowed;
    }

    .settings-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding-top: 8px;
        border-top: 1px solid rgba(138, 92, 246, 0.2);
    }

    .settings-label {
        color: #b9a6ff;
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .controls {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        max-width: 94vw;
        gap: 10px;
        padding: 10px 14px;
        background: rgba(10, 8, 22, 0.45);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }

    .controls button {
        padding: 8px 16px;
        border: 1px solid rgba(138, 92, 246, 0.5);
        border-radius: 6px;
        background: linear-gradient(180deg, rgba(48, 32, 78, 0.85), rgba(18, 12, 32, 0.85));
        color: #e8e2ff;
        font-size: 13px;
        letter-spacing: 0.03em;
        cursor: pointer;
        box-shadow: 0 0 8px rgba(138, 92, 246, 0.25);
        transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.1s ease;
    }

    .controls button:hover {
        border-color: rgba(190, 130, 255, 0.9);
        box-shadow: 0 0 14px rgba(170, 110, 255, 0.5);
    }

    .controls button:active {
        transform: scale(0.96);
    }

    .controls button.active {
        border-color: rgba(255, 221, 51, 0.8);
        color: #fff6d8;
        box-shadow: 0 0 14px rgba(255, 221, 51, 0.45);
    }

    /* Mobile: fewer, smaller buttons (Central Mass moved into the settings panel) keep the
       control bar on one row instead of wrapping, and the debug readout moves out of the
       way of the bottom UI cluster entirely. */
    @media (max-width: 600px) {
        .controls {
            gap: 6px;
            padding: 6px 8px;
            bottom: 12px;
            flex-wrap: nowrap;
            max-width: 96vw;
        }

        .controls button {
            padding: 6px 10px;
            font-size: 11px;
            flex: 0 1 auto;
            white-space: nowrap;
        }

        .settings-panel {
            bottom: 62px;
            max-width: 90vw;
            padding: 10px 14px;
        }

        .debug-panel {
            top: 16px;
            left: 16px;
            bottom: auto;
            padding: 8px 12px;
            font-size: 11px;
        }
    }
</style>
