<template>
    <div class="simulation">
        <canvas ref="canvas"></canvas>
        <div class="debug-panel" v-if="debugPanelVisible">
            <div>v{{ version }}</div>
            <div class="debug-title">Performance</div>
            <div>fps: {{ fps.toFixed(1) }}</div>
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
            <label class="settings-row">
                <input type="checkbox" :checked="mergingEnabled" @change="toggleMerging()" />
                Enable Merging
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="debugPanelVisible" />
                Show Debug Info
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="texturesEnabled" />
                Enable Textures
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

    export default {
        data() {
            return {
                version: constants.VERSION,
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
                potentialEnergy: 0
            };
        },

        created() {
            this.worker = null;
            // A hidden, canvas-less p5 instance kept alive purely to generate the nebula
            // background (see generateBackgroundBitmap) - the actual simulation canvas is
            // owned by the worker via OffscreenCanvas, so nothing here ever calls
            // createCanvas()/draw() on this instance.
            this.bgSketch = null;
            this.resizeTimer = null;
        },

        mounted() {
            this.bgSketch = new p5((sketch) => {
                sketch.setup = () => {
                    sketch.noCanvas();
                    sketch.noLoop();
                };
            });

            this.startSimulation();
            window.addEventListener('resize', this.handleResize);
        },

        beforeUnmount() {
            window.removeEventListener('resize', this.handleResize);
            clearTimeout(this.resizeTimer);
            this.worker?.terminate();
            this.bgSketch?.remove();
        },

        methods: {
            /**
             * Renders the nebula backdrop into an offscreen p5.Graphics buffer and hands
             * back a transferable ImageBitmap - the only piece of rendering still done via
             * p5 (it needs Perlin noise, which isn't worth reimplementing), kept off the
             * hot path since it only happens at startup and on resize, not every frame.
             */
            async generateBackgroundBitmap(width, height) {
                const graphics = createNebulaBackground(this.bgSketch, width, height);
                const bitmap = await createImageBitmap(graphics.canvas);
                graphics.remove();
                return bitmap;
            },

            async startSimulation() {
                const canvasEl = this.$refs.canvas;
                const width = window.innerWidth;
                const height = window.innerHeight;
                canvasEl.width = width;
                canvasEl.height = height;

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
                }, [offscreenCanvas, backgroundBitmap]);
            },

            resetSim() {
                this.worker?.postMessage({ type: 'resetSim', centralMassEnabled: this.centralMassEnabled });
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
