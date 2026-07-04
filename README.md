# Nebula

Genesis of a new Galaxy.. control its evolution and become the ultimate celestial architect



## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Type-Check, Compile and Minify for Production

```sh
npm run build
```


### Layout

```
src/lib/
  constants.ts
  sim/
    colors.ts            mass -> RGB gradient
    surfaceFeatures.ts    crater/cloud/flare/ring generation
    explosion.ts          collision-flash class
    particleRender.ts     drawing one body to canvas
    nebulaBackground.ts   static starfield/nebula backdrop
    particleSystem.ts     the typed-array store + kick/drift/reset/COM/copy
    quadtree.ts           Barnes-Hut tree: flat typed-array nodes, iterative
                          next-pointer traversal, deferred bottom-up COM propagation
    gravity.ts            tree-based force accumulation
    energy.ts             kinetic/potential energy for the debug panel
    merge.ts              merge detection/resolution + compaction (when enabled)
    collide.ts            elastic-bounce collision response (when merging is disabled)
    spawn.ts              initial placement + angular-momentum setup
    simulation.ts         public facade the simulation worker imports
src/workers/
  simulation.worker.ts    owns physics + rendering off the main thread
```

### Threading model

Physics (kick/drift/merge/gravity) and rendering both run inside `simulation.worker.ts`,
not on the main thread. `Particles.vue` transfers the visible `<canvas>` to the worker via
`transferControlToOffscreen()` once at startup, so the worker draws directly to it every
tick - the main thread never touches simulation state or canvas pixels again after that,
only forwarding settings changes in (`postMessage`) and receiving small, infrequent
debug-panel stats back out. This keeps the main thread free for UI/input regardless of
particle count.

There's no `SharedArrayBuffer` involved: that would need `Cross-Origin-Opener-Policy`/
`Cross-Origin-Embedder-Policy` response headers to enable cross-origin isolation, which
GitHub Pages has no way to set. Instead, all simulation state lives entirely inside the
worker, so there's no per-frame data to transfer either direction.

The nebula background (starfield/clouds/galaxies) is the one piece still rendered via p5
(it needs Perlin noise) - generated on the main thread at startup and on resize, then
handed to the worker as a transferable `ImageBitmap` it just blits each frame. Everything
else the worker needs from a "p5 instance" (random/frameCount/fill/ellipse/drawingContext)
is satisfied by a small shim object in `simulation.worker.ts`, so `src/lib/sim/*` runs
completely unmodified in the worker.

Requires `OffscreenCanvas` + `transferControlToOffscreen()` support (Chrome/Edge 69+,
Firefox 105+, Safari 16.4+) - no fallback path for older browsers.