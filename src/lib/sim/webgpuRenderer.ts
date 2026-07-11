// WebGPU rendering backend - an alternative to particleRender.ts's Canvas2D drawing,
// used when the browser supports WebGPU and the user has opted into it. Trades the full
// textured look (craters/clouds/rings/sun flares - a lot of per-particle Canvas2D
// gradient draws) for flat-colored circles rendered via one instanced GPU draw call for
// the whole swarm at once, which is where WebGPU's actual advantage over Canvas2D shows
// up: thousands of tiny per-particle draw calls replaced by a single draw() with an
// instance count, executed in parallel on the GPU instead of one at a time on the CPU.
//
// A canvas can only ever be given ONE rendering context type for its lifetime -
// getContext('2d') and getContext('webgpu') are mutually exclusive, permanently, on the
// same <canvas>. Switching backends at runtime therefore means swapping in a brand new
// canvas element (see Particles.vue) rather than reconfiguring this one - this module
// only ever deals with a canvas that's already been dedicated to WebGPU.
import { densityRamp } from './colors.ts';
import constants from '../constants.ts';

const PARTICLE_FLOATS_PER_INSTANCE = 7; // posX, posY, radius, r, g, b, a
const EXPLOSION_FLOATS_PER_INSTANCE = 7; // posX, posY, radius, r, g, b, a

const BACKGROUND_SHADER = /* wgsl */ `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    out.uv = uvs[vertexIndex];
    return out;
}

@group(0) @binding(0) var bgSampler: sampler;
@group(0) @binding(1) var bgTexture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(bgTexture, bgSampler, in.uv);
}
`;

// Shared by both particles and explosions: a camera-offset instanced quad, expanded to a
// circle in the fragment shader by discarding anything outside radius 1 in local space.
// Particles render fully opaque; explosions additionally fade toward the edge and mix
// toward a white-hot center, approximating the multi-stop radial gradient the Canvas2D
// version uses without needing a second texture lookup.
function circleShader(fadeAtEdge) {
    const fragmentBody = fadeAtEdge
        ? `
    let coreMix = 1.0 - smoothstep(0.0, 0.35, dist);
    let rgb = mix(in.color.rgb, vec3<f32>(1.0, 1.0, 0.98), coreMix);
    let alpha = in.color.a * (1.0 - smoothstep(0.55, 1.0, dist));
    return vec4<f32>(rgb, alpha);`
        : `
    return in.color;`;

    return /* wgsl */ `
struct Camera {
    offset: vec2<f32>,
    canvasSize: vec2<f32>,
    zoom: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) localPos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @location(0) instancePos: vec2<f32>,
    @location(1) instanceRadius: f32,
    @location(2) instanceColor: vec4<f32>,
) -> VertexOutput {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertexIndex];
    // camera.offset already resolves to (screenCenter - anchor*zoom + pan) on the CPU side
    // (see simulation.worker.ts's cpuTick/gpuTick) - scaling instancePos and the corner's
    // own radius offset by zoom here, then adding that offset unscaled, is what makes
    // distances from the camera anchor (not the world origin) grow/shrink with zoom while
    // still landing the anchor itself exactly on screen center + pan regardless of zoom.
    let worldPos = (instancePos + corner * instanceRadius) * camera.zoom + camera.offset;
    let ndcX = (worldPos.x / camera.canvasSize.x) * 2.0 - 1.0;
    let ndcY = 1.0 - (worldPos.y / camera.canvasSize.y) * 2.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    out.localPos = corner;
    out.color = instanceColor;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.localPos);
    if (dist > 1.0) {
        discard;
    }
    ${fragmentBody}
}
`;
}

const CROSSHAIR_SHADER = /* wgsl */ `
struct Camera {
    offset: vec2<f32>,
    canvasSize: vec2<f32>,
    zoom: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
// Already the fully-resolved SCREEN position (screenCenter + cameraPan - see
// simulation.worker.ts) the camera anchor (COM or fixed world center) maps to, which is
// exactly where this marker always belongs regardless of zoom - the anchor's own
// distance from itself is zero, so it's invariant to the camera.offset/zoom math the
// particle shaders use. This deliberately does NOT add camera.offset or scale by
// camera.zoom - the crosshair's 8px arm length stays a constant on-screen size instead of
// growing/shrinking with zoom, matching the Canvas2D path's compensated armHalf.
@group(0) @binding(1) var<uniform> center: vec2<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    var offsets = array<vec2<f32>, 4>(
        vec2<f32>(-8.0, 0.0), vec2<f32>(8.0, 0.0),
        vec2<f32>(0.0, -8.0), vec2<f32>(0.0, 8.0),
    );
    let worldPos = center + offsets[vertexIndex];
    let ndcX = (worldPos.x / camera.canvasSize.x) * 2.0 - 1.0;
    let ndcY = 1.0 - (worldPos.y / camera.canvasSize.y) * 2.0;
    return vec4<f32>(ndcX, ndcY, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`;

/** True if this environment can plausibly render with WebGPU at all. */
export async function isWebGPUSupported() {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}

function growInstanceBuffer(device, oldBuffer, neededFloats, usage) {
    const neededBytes = neededFloats * 4;
    if (oldBuffer && oldBuffer.size >= neededBytes) {
        return oldBuffer;
    }
    if (oldBuffer) {
        oldBuffer.destroy();
    }
    // Round up generously so growth doesn't happen every frame as particle/explosion
    // counts fluctuate by one or two.
    const capacityBytes = Math.max(neededBytes * 2, 4096);
    return device.createBuffer({
        size: capacityBytes,
        usage,
        mappedAtCreation: false,
    });
}

/**
 * Sets up the WebGPU device, pipelines, and per-frame resources for the given
 * (already-dedicated-to-WebGPU) canvas. Returns null if anything in the chain fails
 * (adapter/device request, pipeline compilation) so the caller can fall back to
 * Canvas2D instead of leaving the app in a half-initialized state.
 *
 * `existingDevice` lets the GPU-physics mode (webgpuSim.ts) share one device between
 * compute and rendering - required for the render pipeline to read the sim's position
 * buffer directly (GPU buffers are never shareable across devices). A renderer built on
 * a caller-owned device won't destroy it in destroyWebGPURenderer.
 */
export async function createWebGPURenderer(canvas, existingDevice = null) {
    if (!navigator.gpu) {
        return null;
    }

    let adapter;
    let device = existingDevice;
    try {
        if (!device) {
            adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return null;
            device = await adapter.requestDevice();
        }
    } catch {
        return null;
    }

    const context = canvas.getContext('webgpu');
    if (!context) {
        return null;
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const cameraBuffer = device.createBuffer({
        // vec2 offset + vec2 canvasSize + f32 zoom = 20 bytes, rounded up to WGSL's
        // 8-byte struct alignment (16 + 4 -> 24) and then to 32 for a comfortable margin
        // - some implementations validate uniform buffer bindings against extra padding.
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const backgroundSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const backgroundModule = device.createShaderModule({ code: BACKGROUND_SHADER });
    const backgroundPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: backgroundModule, entryPoint: 'vs_main' },
        fragment: { module: backgroundModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
    });

    const circleBufferLayout = [
        {
            arrayStride: PARTICLE_FLOATS_PER_INSTANCE * 4,
            stepMode: 'instance',
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x2' }, // pos
                { shaderLocation: 1, offset: 8, format: 'float32' }, // radius
                { shaderLocation: 2, offset: 12, format: 'float32x4' }, // color
            ],
        },
    ];

    const particleModule = device.createShaderModule({ code: circleShader(false) });
    const particlePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: particleModule, entryPoint: 'vs_main', buffers: circleBufferLayout },
        fragment: { module: particleModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
    });

    const explosionModule = device.createShaderModule({ code: circleShader(true) });
    const explosionPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: explosionModule, entryPoint: 'vs_main', buffers: circleBufferLayout },
        fragment: {
            module: explosionModule,
            entryPoint: 'fs_main',
            targets: [{
                format,
                blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                },
            }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    const crosshairCenterBuffer = device.createBuffer({
        size: 16, // vec2<f32> is 8 bytes, but padded to 16 - some implementations
        // validate uniform buffer bindings against a 16-byte minimum regardless of the
        // WGSL type's own natural size.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const crosshairModule = device.createShaderModule({ code: CROSSHAIR_SHADER });
    const crosshairPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: crosshairModule, entryPoint: 'vs_main' },
        fragment: { module: crosshairModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'line-list' },
    });

    const particleBindGroup = device.createBindGroup({
        layout: particlePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
    });
    const explosionBindGroup = device.createBindGroup({
        layout: explosionPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
    });
    const crosshairBindGroup = device.createBindGroup({
        layout: crosshairPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cameraBuffer } },
            { binding: 1, resource: { buffer: crosshairCenterBuffer } },
        ],
    });

    const renderer = {
        device,
        ownsDevice: !existingDevice,
        context,
        format,
        canvas,
        // GPU-physics mode (see attachSimBuffers): a pipeline that reads particle
        // positions straight from webgpuSim.ts's storage buffer - zero per-frame copies.
        simPipeline: null,
        simBindGroup: null,
        simCount: 0,
        cameraBuffer,
        crosshairCenterBuffer,
        backgroundSampler,
        backgroundPipeline,
        particlePipeline,
        particleBindGroup,
        explosionPipeline,
        explosionBindGroup,
        crosshairPipeline,
        crosshairBindGroup,
        backgroundTexture: null,
        backgroundBindGroup: null,
        particleBuffer: null,
        particleData: new Float32Array(0),
        explosionBuffer: null,
        explosionData: new Float32Array(0),
    };

    return renderer;
}

/** Call whenever the canvas's pixel size changes - WebGPU needs the context reconfigured. */
export function resizeWebGPURenderer(renderer, width, height) {
    renderer.canvas.width = width;
    renderer.canvas.height = height;
    renderer.context.configure({ device: renderer.device, format: renderer.format, alphaMode: 'opaque' });
}

/** Uploads the nebula background bitmap as a sampled texture, replacing any previous one. */
export function setWebGPUBackground(renderer, bitmap) {
    if (renderer.backgroundTexture) {
        renderer.backgroundTexture.destroy();
    }
    renderer.backgroundTexture = renderer.device.createTexture({
        size: [bitmap.width, bitmap.height],
        format: renderer.format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    renderer.device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture: renderer.backgroundTexture },
        [bitmap.width, bitmap.height]
    );
    renderer.backgroundBindGroup = renderer.device.createBindGroup({
        layout: renderer.backgroundPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: renderer.backgroundSampler },
            { binding: 1, resource: renderer.backgroundTexture.createView() },
        ],
    });
}

function fillParticleData(renderer, system, densityColors) {
    const neededFloats = Math.max(system.count, 1) * PARTICLE_FLOATS_PER_INSTANCE;
    if (renderer.particleData.length < neededFloats) {
        renderer.particleData = new Float32Array(neededFloats);
    }
    const data = renderer.particleData;
    for (let i = 0; i < system.count; i++) {
        const base = i * PARTICLE_FLOATS_PER_INSTANCE;
        data[base] = system.posX[i];
        data[base + 1] = system.posY[i];
        data[base + 2] = system.radius[i];
        if (densityColors) {
            const [dr, dg, db] = densityRamp(system.density[i]);
            data[base + 3] = dr / 255;
            data[base + 4] = dg / 255;
            data[base + 5] = db / 255;
        } else {
            data[base + 3] = system.colorR[i] / 255;
            data[base + 4] = system.colorG[i] / 255;
            data[base + 5] = system.colorB[i] / 255;
        }
        data[base + 6] = 1;
    }
    renderer.particleBuffer = growInstanceBuffer(
        renderer.device,
        renderer.particleBuffer,
        neededFloats,
        GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );
    renderer.device.queue.writeBuffer(renderer.particleBuffer, 0, data.buffer, 0, system.count * PARTICLE_FLOATS_PER_INSTANCE * 4);
}

function fillExplosionData(renderer, explosions) {
    const count = explosions.length;
    const neededFloats = Math.max(count, 1) * EXPLOSION_FLOATS_PER_INSTANCE;
    if (renderer.explosionData.length < neededFloats) {
        renderer.explosionData = new Float32Array(neededFloats);
    }
    const data = renderer.explosionData;
    for (let i = 0; i < count; i++) {
        const explosion = explosions[i];
        const t = explosion.age / explosion.maxAge;
        const growth = 1 - Math.pow(1 - Math.min(t / 0.25, 1), 3);
        const radius = explosion.peakRadius * growth;
        const alpha = Math.pow(1 - t, 2);
        const base = i * EXPLOSION_FLOATS_PER_INSTANCE;
        data[base] = explosion.x;
        data[base + 1] = explosion.y;
        data[base + 2] = radius;
        data[base + 3] = explosion.color[0] / 255;
        data[base + 4] = explosion.color[1] / 255;
        data[base + 5] = explosion.color[2] / 255;
        data[base + 6] = alpha;
    }
    if (count > 0) {
        renderer.explosionBuffer = growInstanceBuffer(
            renderer.device,
            renderer.explosionBuffer,
            neededFloats,
            GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        );
        renderer.device.queue.writeBuffer(renderer.explosionBuffer, 0, data.buffer, 0, count * EXPLOSION_FLOATS_PER_INSTANCE * 4);
    }
}

/**
 * Draws one full frame: background, particles, explosions, optional crosshair - the
 * WebGPU-backend equivalent of simulation.ts's displayAll + the worker's own
 * explosion/crosshair drawing. Textures aren't implemented on this path (see this file's
 * header comment) - particles always render as flat-colored circles here regardless of
 * the texturesEnabled setting. `densityColors` swaps each particle's own color for
 * colors.ts's densityRamp over its current local crowding (system.density[i]).
 */
export function renderWebGPUFrame(renderer, system, explosions, camera, crosshairVisible, densityColors = false) {
    const device = renderer.device;

    const cameraData = new Float32Array([camera.offsetX, camera.offsetY, renderer.canvas.width, renderer.canvas.height, camera.zoom]);
    device.queue.writeBuffer(renderer.cameraBuffer, 0, cameraData.buffer);

    fillParticleData(renderer, system, densityColors);
    if (explosions.length > 0) {
        fillExplosionData(renderer, explosions);
    }

    const encoder = device.createCommandEncoder();
    const view = renderer.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });

    if (renderer.backgroundBindGroup) {
        pass.setPipeline(renderer.backgroundPipeline);
        pass.setBindGroup(0, renderer.backgroundBindGroup);
        pass.draw(4);
    }

    if (system.count > 0) {
        pass.setPipeline(renderer.particlePipeline);
        pass.setBindGroup(0, renderer.particleBindGroup);
        pass.setVertexBuffer(0, renderer.particleBuffer);
        pass.draw(4, system.count);
    }

    if (explosions.length > 0) {
        pass.setPipeline(renderer.explosionPipeline);
        pass.setBindGroup(0, renderer.explosionBindGroup);
        pass.setVertexBuffer(0, renderer.explosionBuffer);
        pass.draw(4, explosions.length);
    }

    if (crosshairVisible) {
        const centerData = new Float32Array([camera.crosshairX, camera.crosshairY]);
        device.queue.writeBuffer(renderer.crosshairCenterBuffer, 0, centerData.buffer);
        pass.setPipeline(renderer.crosshairPipeline);
        pass.setBindGroup(0, renderer.crosshairBindGroup);
        pass.draw(4);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
}

/**
 * Builds the GPU-physics render path: an instanced circle pipeline whose per-particle
 * position comes straight from the sim's storage buffer via instance_index (radius/color
 * from the sim's static color buffer) - no vertex-buffer fill, no CPU copy, ever. Called
 * once per sim (re)creation; the particle count and small-particle rendering constants
 * are baked into the shader.
 *
 * At the particle counts this path exists for (up to 1M), individual radii drop well
 * below a pixel (radius = sqrt(MAX_MASS/count) ~ 0.02px at 1M) - rasterizing those
 * faithfully would draw nothing. Rendered size clamps to half a pixel and blending is
 * additive, so dense regions read as a brightening density field (the same practical
 * choice the native renderer makes via its metaball field accumulation, reduced to its
 * cheapest form).
 */
/**
 * Generates a WGSL densityRamp(t) function from constants.ts's DENSITY_COLOR_STOPS,
 * matching colors.ts's interpolateColorStops exactly (evenly-spaced segments across
 * 0..1, clamped, linear-interpolated within whichever segment t falls into) - the GPU
 * render pipeline can't call back into JS per-pixel, so this compiles the current stops
 * straight into the shader source instead. Regenerated fresh every time the GPU sim
 * (re)builds (attachSimBuffers runs once per creation), so editing DENSITY_COLOR_STOPS
 * takes effect on the next Restart/solver switch, not live mid-run.
 */
function buildDensityRampWGSL(stops) {
    const segmentCount = stops.length - 1;
    const stopVecs = stops
        .map(([r, g, b]) => `vec3<f32>(${r / 255}, ${g / 255}, ${b / 255})`)
        .join(', ');
    return /* wgsl */ `
fn densityRamp(t: f32) -> vec3<f32> {
    var stops = array<vec3<f32>, ${stops.length}>(${stopVecs});
    let clampedT = clamp(t, 0.0, 1.0);
    let scaled = clampedT * ${segmentCount}.0;
    let idx = min(i32(floor(scaled)), ${segmentCount - 1});
    let localT = scaled - f32(idx);
    return mix(stops[idx], stops[idx + 1], localT);
}
`;
}

export function attachSimBuffers(renderer, sim) {
    const device = renderer.device;
    const alpha = sim.count > 100000 ? 0.3 : 1.0;
    const densityRampWGSL = buildDensityRampWGSL(constants.DENSITY_COLOR_STOPS);
    const shader = /* wgsl */ `
struct Camera {
    offset: vec2<f32>,
    canvasSize: vec2<f32>,
    zoom: f32,
};
struct DensityMode {
    // Only .x is meaningful - std140-style uniform layout pads to 16 bytes regardless.
    mode: i32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> colorRadius: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> density: array<f32>;
@group(0) @binding(4) var<uniform> densityMode: DensityMode;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) localPos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Generated from constants.ts's DENSITY_COLOR_STOPS by buildDensityRampWGSL() above -
// see that function's comment for why this can't just call colors.ts's JS densityRamp
// directly.
${densityRampWGSL}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) inst: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertexIndex];
    let cr = colorRadius[inst];
    let radius = max(cr.w, 0.5);
    // Same zoom-around-anchor convention as circleShader() above - see that shader's
    // comment for the full derivation.
    let worldPos = (particlePos[inst] + corner * radius) * camera.zoom + camera.offset;
    let ndcX = (worldPos.x / camera.canvasSize.x) * 2.0 - 1.0;
    let ndcY = 1.0 - (worldPos.y / camera.canvasSize.y) * 2.0;
    var out: VertexOutput;
    out.position = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
    out.localPos = corner;
    let baseColor = select(cr.rgb, densityRamp(density[inst]), densityMode.mode != 0);
    out.color = vec4<f32>(baseColor, ${alpha});
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (length(in.localPos) > 1.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb * in.color.a, in.color.a);
}
`;
    const module = device.createShaderModule({ code: shader });
    renderer.simPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
            module,
            entryPoint: 'fs_main',
            targets: [{
                format: renderer.format,
                blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                },
            }],
        },
        primitive: { topology: 'triangle-strip' },
    });
    if (!renderer.simDensityModeBuffer) {
        renderer.simDensityModeBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    renderer.simBindGroup = device.createBindGroup({
        layout: renderer.simPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: renderer.cameraBuffer } },
            { binding: 1, resource: { buffer: sim.posBuf } },
            { binding: 2, resource: { buffer: sim.colorRadiusBuf } },
            { binding: 3, resource: { buffer: sim.densityOutBuf } },
            { binding: 4, resource: { buffer: renderer.simDensityModeBuffer } },
        ],
    });
    renderer.simCount = sim.count;
}

/**
 * Draws one GPU-physics frame straight from the sim's buffers - the counterpart to
 * renderWebGPUFrame for when the particle state lives on the GPU (no explosions: the
 * GPU pipeline is bounce-only, merges never happen there). `densityColors` colors every
 * particle by its live GPU-computed contact-derived density (webgpuSim.ts's
 * densityOutBuf) via the shader's densityRamp instead of its own color.
 */
export function renderWebGPUSimFrame(renderer, camera, crosshairVisible, densityColors = false) {
    if (!renderer.simPipeline) return;
    const device = renderer.device;

    const cameraData = new Float32Array([camera.offsetX, camera.offsetY, renderer.canvas.width, renderer.canvas.height, camera.zoom]);
    device.queue.writeBuffer(renderer.cameraBuffer, 0, cameraData.buffer);
    device.queue.writeBuffer(renderer.simDensityModeBuffer, 0, new Int32Array([densityColors ? 1 : 0]));

    const encoder = device.createCommandEncoder();
    const view = renderer.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });

    if (renderer.backgroundBindGroup) {
        pass.setPipeline(renderer.backgroundPipeline);
        pass.setBindGroup(0, renderer.backgroundBindGroup);
        pass.draw(4);
    }

    pass.setPipeline(renderer.simPipeline);
    pass.setBindGroup(0, renderer.simBindGroup);
    pass.draw(4, renderer.simCount);

    if (crosshairVisible) {
        const centerData = new Float32Array([camera.crosshairX, camera.crosshairY]);
        device.queue.writeBuffer(renderer.crosshairCenterBuffer, 0, centerData.buffer);
        pass.setPipeline(renderer.crosshairPipeline);
        pass.setBindGroup(0, renderer.crosshairBindGroup);
        pass.draw(4);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
}

/** Releases GPU resources - call when switching away from this renderer. */
export function destroyWebGPURenderer(renderer) {
    if (renderer.backgroundTexture) renderer.backgroundTexture.destroy();
    if (renderer.particleBuffer) renderer.particleBuffer.destroy();
    if (renderer.explosionBuffer) renderer.explosionBuffer.destroy();
    if (renderer.simDensityModeBuffer) renderer.simDensityModeBuffer.destroy();
    renderer.cameraBuffer.destroy();
    renderer.crosshairCenterBuffer.destroy();
    // A shared device belongs to the GPU-physics sim's lifecycle, not this renderer's -
    // destroying it here would kill the compute pipeline mid-flight.
    if (renderer.ownsDevice) {
        renderer.device.destroy();
    }
}
