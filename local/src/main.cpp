// Native C++/OpenGL port of the Nebula N-body sim - non-merging collisions (bounce) plus
// the optional single fixed central mass, no textures/explosion-flash effects (see the
// header comments in collide.h/spawn.h/renderer.h for what was deliberately left out and
// why). Built to answer one question: how does this perform as a local executable versus
// the browser/WASM version - see README-less local/build.sh for how to compile it.
//
// Usage: nebula_native [particleCount] [--central-mass] [--pm] [--pm-grid N] [--gpu]
//   --pm         use Particle-Mesh gravity (see pm_gravity.h) instead of the Barnes-Hut
//                tree - a different gravity *model* (2D Poisson solve, ~1/r force law),
//                not just a faster path to the same physics; see pm_gravity.h's header
//                comment. The CPU mesh solve uses isolated (free-space) boundaries via
//                zero-padded FFTs - no periodic image forces (see pm_gravity.h's header
//                for the anisotropic-collapse artifact the old periodic solve caused).
//                Positions are never wrapped - a particle that exits the window keeps
//                going.
//   --pm-grid N  PM grid resolution per axis, rounded up to a power of two (default 256).
//   --gpu        run the ENTIRE physics pipeline (integration, PM gravity, P3M correction,
//                collision) as OpenGL 4.3 compute shaders - see gpu_sim.h, including where
//                its collision semantics deliberately differ from the CPU path. Implies
//                --pm, and lifts the particle cap to 4,000,000 (positions never leave GPU
//                memory, so CPU-side per-particle costs vanish).
//   --density-colors  color particles by local crowding instead of their own color: blue
//                when isolated, red at mid density, climbing through yellow to white in
//                packed cores - a heat-map over the same per-particle density signal that
//                already drives the blur effect (renderer.h's densityRamp).
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include "particle_system.h"
#include "tree.h"
#include "gravity.h"
#include "collide.h"
#include "spawn.h"
#include "renderer.h"
#include "ui.h"
#include "pm_gravity.h"
#include "gpu_sim.h"
#include "constants.h"

inline int32_t roundUpPow2(int32_t v) {
    int32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Kept as one long string (not scattered per-flag comments elsewhere) so --help output
// and the flags actually parsed in main() can't silently drift apart - if a flag's
// behavior changes, this text and the parsing loop right below it are the two places to
// update together.
inline void printUsage(const char* argv0) {
    printf(
        "Nebula native - N-body gravity/collision simulation (C++/OpenGL port)\n"
        "\n"
        "Usage: %s [particleCount] [options]\n"
        "\n"
        "  particleCount         Number of swarm particles to spawn (default 2500).\n"
        "                        Capped at %d normally, 4,000,000 under --gpu.\n"
        "\n"
        "Options:\n"
        "  --central-mass        Add one additional fixed particle at the exact window\n"
        "                        center, holding a large share of total mass - the swarm\n"
        "                        orbits/accretes around it instead of just each other.\n"
        "  --pm                  Use Particle-Mesh gravity (FFT Poisson solve, isolated\n"
        "                        boundaries) instead of the default Barnes-Hut tree - a\n"
        "                        different force law (~1/r, true 2D), not just a faster\n"
        "                        path to the same physics. See pm_gravity.h.\n"
        "  --pm-grid N           PM mesh resolution per axis, rounded up to a power of\n"
        "                        two (default 256, auto-scaled under --gpu unless this is\n"
        "                        given explicitly). Higher = finer short-range accuracy,\n"
        "                        quadratically more FFT cost.\n"
        "  --gpu                 Run the entire physics pipeline (integration, PM\n"
        "                        gravity, P3M correction, collision) as OpenGL 4.3\n"
        "                        compute shaders instead of the CPU. Implies --pm.\n"
        "  --density-colors      Color particles by local crowding (blue=isolated,\n"
        "                        red=mid-density, yellow/white=packed cores) instead of\n"
        "                        their own flat color.\n"
        "  --throttle            Cap the frame rate to roughly 100fps instead of running\n"
        "                        uncapped - useful for eyeballing behavior without\n"
        "                        pegging a CPU/GPU core at full tilt.\n"
        "  -h, --help            Show this message and exit.\n"
        "\n"
        "Controls: scroll to zoom (toward cursor), click-drag to pan, on-screen buttons\n"
        "to stop/resume, restart, and open settings.\n",
        argv0, constants::MAX_PARTICLES);
}

constexpr float MIN_ZOOM = 0.1f;
constexpr float MAX_ZOOM = 20.f;

// Camera state, reached from inside GLFW's plain-C-function-pointer callbacks via
// glfwSetWindowUserPointer/glfwGetWindowUserPointer (callbacks take no user-data parameter
// of their own). offsetX/Y is the world point currently mapped to the center of the screen
// (renderer.h's cameraOffset uniform) - panning (click-and-drag) moves it, zooming scales
// distances from it.
struct CameraState {
    float zoom = 1.f;
    float offsetX = 0.f, offsetY = 0.f;
    float windowWidth = 0.f, windowHeight = 0.f; // needed to convert cursor pos -> world pos below
    bool dragging = false;
    double lastMouseX = 0.0, lastMouseY = 0.0;
};

// Each wheel notch multiplies zoom by 1.1 (scrolling up/away zooms in) rather than adding a
// fixed amount, so it feels like a constant-percentage zoom step regardless of the current
// zoom level. Zooms toward the cursor, not the window center: the world point currently
// under the mouse is computed at the OLD zoom, then offsetX/Y is solved so that same world
// point still lands under the cursor at the NEW zoom - the same "screenPos = (worldPos -
// cameraOffset)*zoom + screenCenter" relationship renderer.h's shader uses, just inverted
// and re-solved for cameraOffset instead of screenPos.
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    CameraState* cam = (CameraState*)glfwGetWindowUserPointer(window);
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);

    float screenCenterX = cam->windowWidth * 0.5f;
    float screenCenterY = cam->windowHeight * 0.5f;
    float worldUnderMouseX = cam->offsetX + ((float)mouseX - screenCenterX) / cam->zoom;
    float worldUnderMouseY = cam->offsetY + ((float)mouseY - screenCenterY) / cam->zoom;

    cam->zoom *= powf(1.1f, (float)yoffset);
    cam->zoom = std::max(MIN_ZOOM, std::min(cam->zoom, MAX_ZOOM));

    cam->offsetX = worldUnderMouseX - ((float)mouseX - screenCenterX) / cam->zoom;
    cam->offsetY = worldUnderMouseY - ((float)mouseY - screenCenterY) / cam->zoom;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    CameraState* cam = (CameraState*)glfwGetWindowUserPointer(window);
    if (action == GLFW_PRESS) {
        cam->dragging = true;
        glfwGetCursorPos(window, &cam->lastMouseX, &cam->lastMouseY);
    } else if (action == GLFW_RELEASE) {
        cam->dragging = false;
    }
}

// Dragging moves the camera opposite to the cursor delta, scaled by 1/zoom (screen pixels ->
// world units) - so the world point that was under the cursor at drag start stays under it
// as the drag continues, the standard "grab and move the canvas" feel.
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    CameraState* cam = (CameraState*)glfwGetWindowUserPointer(window);
    if (cam->dragging) {
        float dx = (float)(xpos - cam->lastMouseX);
        float dy = (float)(ypos - cam->lastMouseY);
        cam->offsetX -= dx / cam->zoom;
        cam->offsetY -= dy / cam->zoom;
    }
    cam->lastMouseX = xpos;
    cam->lastMouseY = ypos;
}

int main(int argc, char** argv) {
    int32_t totalParticles = 2500;
    bool includeCentralMass = false;
    bool usePM = false;
    bool useGpu = false;
    bool useThrottle = false;
    bool densityColors = false;
    int32_t pmGridSize = 256;
    bool pmGridExplicit = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--central-mass") == 0) {
            includeCentralMass = true;
        } else if (strcmp(argv[i], "--pm") == 0) {
            usePM = true;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            useGpu = true;
            usePM = true; // GPU pipeline is PM-based; there is no GPU tree
        } else if (strcmp(argv[i], "--pm-grid") == 0 && i + 1 < argc) {
            pmGridSize = atoi(argv[++i]);
            pmGridExplicit = true;
        } else if (strcmp(argv[i], "--throttle") == 0) {
            useThrottle = true;
        } else if (strcmp(argv[i], "--density-colors") == 0) {
            densityColors = true;
        } else {
            totalParticles = atoi(argv[i]);
            if (totalParticles < 1) totalParticles = 1;
        }
    }
    // GPU mode lifts the CPU path's cap: per-particle state lives in GPU buffers and the
    // only CPU arrays are the one-time spawn (constants::MAX_PARTICLES sizes CPU physics
    // scratch that GPU mode never allocates).
    int32_t particleCap = useGpu ? 4000000 : constants::MAX_PARTICLES;
    if (totalParticles > particleCap - (includeCentralMass ? 1 : 0)) {
        totalParticles = particleCap - (includeCentralMass ? 1 : 0);
    }
    // GPU mode auto-scales the mesh to the particle count unless --pm-grid was given
    // explicitly: P3M pair count per particle ~ (rCut^2 * areal density), and rCut tracks
    // cell size - so cell size has to shrink as density grows or the short-range pass
    // degenerates into an O(n^2)-flavored wall (measured: 1M particles on the default 256
    // grid spent ~700ms/frame in P3M pairs, most of them dropped by cell overflow anyway;
    // a finer mesh resolves those scales itself and leaves P3M a bounded per-cell load).
    if (useGpu && !pmGridExplicit) {
        pmGridSize = roundUpPow2((int32_t)sqrtf((float)totalParticles * 2.f));
        pmGridSize = std::max(256, pmGridSize);
    }
    pmGridSize = std::max(32, std::min(roundUpPow2(pmGridSize), 2048));

    char gravityLabel[32];
    if (usePM) {
        snprintf(gravityLabel, sizeof(gravityLabel), "%sPM %dX%d", useGpu ? "GPU " : "", pmGridSize, pmGridSize);
    } else {
        snprintf(gravityLabel, sizeof(gravityLabel), "TREE");
    }

    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    // Compute shaders need 4.3; the CPU path keeps requesting 3.3 so it still runs on
    // older drivers.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, useGpu ? 4 : 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int windowWidth = 1600;
    const int windowHeight = 900;
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Nebula (native)", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // uncapped - this benchmark wants to measure real frame time, not get clamped to vsync

    CameraState cam;
    cam.windowWidth = (float)windowWidth;
    cam.windowHeight = (float)windowHeight;
    cam.offsetX = (float)windowWidth * 0.5f;
    cam.offsetY = (float)windowHeight * 0.5f;
    glfwSetWindowUserPointer(window, &cam);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    ParticleSystem sys;
    SpatialTree gravityTree;
    SpatialTree collisionTree;
    SpatialTree p3mTree;
    CollisionCandidates candidates;
    PMGrid pmGrid;
    PMShortRangeTable pmShortRangeTable;
    std::vector<CicWeights> pmWeightsScratch;
    GpuSim gpuSim;

    int32_t capacity = totalParticles + (includeCentralMass ? 1 : 0);
    if (usePM) {
        pmGrid.allocate(pmGridSize, pmGridSize, (float)windowWidth, (float)windowHeight);
        if (!useGpu) p3mTree.allocate(capacity);
        // One-time startup cost (see PMShortRangeTable::calibrate) - runs the actual mesh
        // solve at PM_CALIBRATION_TABLE_SIZE separations on a grid matching production
        // exactly, so the correction is self-consistent with whatever this specific grid
        // resolution/domain/G actually produces. The GPU pipeline uploads and reuses this
        // same CPU-calibrated table - its mesh solve is the same algorithm end-to-end, so
        // the calibration carries over (see gpu_sim.h's header comment).
        float p3mCutoff = constants::PM_P3M_CUTOFF_FACTOR * sqrtf(pmGrid.cellWidth * pmGrid.cellHeight);
        pmShortRangeTable.calibrate(pmGrid.cellWidth, pmGrid.cellHeight,
                                     constants::PM_GRAVITATIONAL_CONSTANT, p3mCutoff, constants::PM_CALIBRATION_TABLE_SIZE);
    } else {
        gravityTree.allocate(capacity);
    }
    if (!useGpu) {
        // CPU physics scratch - never touched in GPU mode, and at GPU-scale particle counts
        // (candidates alone would be capacity * 128 ints - 2GB at 4M) must not be allocated.
        collisionTree.allocate(capacity);
        candidates.allocate(capacity);
    }
    std::vector<float> densityScratch(useGpu ? 0 : capacity, 0.f); // rendering only - see renderer.h's header comment

    std::mt19937 rng(12345);
    spawnParticles(sys, totalParticles, includeCentralMass, (float)windowWidth, (float)windowHeight, rng);

    Renderer renderer;
    renderer.init(capacity);
    renderer.densityColors = densityColors;
    renderer.uploadStaticInstanceData(sys);

    UiRenderer ui;
    ui.init();

    // Prime accX/accY before the first kick - see main.cpp's header comment on step ordering.
    if (useGpu) {
        gpuSim.init(sys, pmGridSize, (float)windowWidth, (float)windowHeight, pmGrid, pmShortRangeTable,
                    constants::PM_GRAVITATIONAL_CONSTANT, constants::PM_PAIR_SOFTENING_FACTOR,
                    constants::GRAVITATIONAL_CONSTANT, constants::GRAVITY_SOFTENING_FACTOR,
                    constants::COLLISION_RESTITUTION, constants::COLLISION_SURFACE_GAP);
        // Prime acc on GPU: run one full substep at dt=0 - integration moves nothing, but
        // the gravity chain fills accBuf so the first real substep's kick has a force.
        gpuSim.substep(0.f, constants::DENSITY_BLUR_THRESHOLD, constants::COLLISION_SURFACE_GAP);
        gpuSim.finishFrame();
    } else if (usePM) {
        computeGravityPM(sys, pmGrid, p3mTree, pmShortRangeTable, constants::PM_GRAVITATIONAL_CONSTANT,
                          constants::PM_PAIR_SOFTENING_FACTOR, pmWeightsScratch, nullptr, constants::GRAVITY_MAX_THREADS);
    } else {
        computeGravity(sys, gravityTree, constants::GRAVITATIONAL_CONSTANT, constants::GRAVITY_SOFTENING_FACTOR,
                       constants::BARNES_HUT_THETA, constants::QUADTREE_MAX_DEPTH, constants::QUADTREE_LEAF_CAPACITY,
                       constants::GRAVITY_MAX_THREADS);
    }

    printf("Nebula native: %d particles%s, gravity=%s, hardware_concurrency=%u\n", totalParticles,
           includeCentralMass ? " + central mass" : "", gravityLabel, std::thread::hardware_concurrency());

    double frameTimeAccumMs = 0.0;
    int32_t frameCount = 0;
    auto lastFpsPrint = std::chrono::steady_clock::now();
    double displayAvgMs = 0.0; // updated once/sec (see the print block below) - drives the on-screen panel too, so its numbers don't flicker every frame
    int32_t displayThreadCount = (int32_t)std::min((unsigned int)std::thread::hardware_concurrency(), (unsigned int)constants::GRAVITY_MAX_THREADS);

    // Per-stage timing breakdown, printed alongside the frame average once/sec - a
    // temporary diagnostic to find the real bottleneck rather than guessing at it.
    double accumKickDriftMs = 0.0, accumCollideMs = 0.0, accumGravityMs = 0.0, accumRenderMs = 0.0;
    double accumCollideBuildMs = 0.0, accumCollideSearchMs = 0.0, accumCollideResolveMs = 0.0;
    int64_t accumTotalCandidates = 0;

    int32_t displaySubsteps = 1; // updated once/sec, drives the debug panel - see the print block below
    float displayComX = 0.f, displayComY = 0.f; // GPU mode refreshes these once/sec (see the stats block)
    std::vector<float> gpuPosReadback;

    // FPS Throttle in milliseconds
    const double frameTime = 10;

    while (!glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::steady_clock::now();

        // Adaptive substepping: a fixed dt=1 kick can badly overshoot a close encounter
        // (see constants.h's SUBSTEP_SAFETY_FACTOR comment) - estimate how many smaller
        // kick-drift-collide-gravity cycles this frame actually needs from the peak
        // acceleration already sitting in accX/accY (computed by last frame's - or the
        // initial priming call's - gravity solve), then run that many, each with a
        // correspondingly smaller dt. Sparse frames stay at substeps=1 (identical cost and
        // behavior to before this feature existed). GPU mode gets the same decision from
        // its 8-byte max-acceleration readback (one frame stale, same as the CPU path's
        // "last solve's acc" semantics).
        float maxAccel;
        if (useGpu) {
            maxAccel = gpuSim.lastMaxAccel;
        } else {
            float maxAccelSq = 0.f;
            for (int32_t i = 0; i < sys.count; i++) {
                float aSq = sys.accX[i] * sys.accX[i] + sys.accY[i] * sys.accY[i];
                if (aSq > maxAccelSq) maxAccelSq = aSq;
            }
            maxAccel = sqrtf(maxAccelSq);
        }
        float representativeRadius = sys.count > 0 ? sys.radius[0] : 1.f; // all swarm particles share one radius - see spawn.h
        int32_t substeps = 1;
        if (maxAccel > 0.f && representativeRadius > 0.f) {
            float needed = sqrtf(maxAccel / (constants::SUBSTEP_SAFETY_FACTOR * representativeRadius));
            substeps = (int32_t)ceilf(needed);
        }
        substeps = std::max(1, std::min(substeps, constants::MAX_SUBSTEPS));
        displaySubsteps = substeps;
        float dt = 1.f / (float)substeps;

        int64_t totalCandidatesThisFrame = 0;

        for (int32_t sub = 0; sub < substeps && useGpu; sub++) {
            auto t0 = std::chrono::steady_clock::now();
            gpuSim.substep(dt, constants::DENSITY_BLUR_THRESHOLD, constants::COLLISION_SURFACE_GAP);
            auto t1 = std::chrono::steady_clock::now();
            accumGravityMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (useGpu) gpuSim.finishFrame();

        for (int32_t sub = 0; sub < substeps && !useGpu; sub++) {
            // Kick-drift-collide-kick ordering, matching the web app's simulation.ts: kick
            // and drift with this substep's already-known acceleration, resolve any
            // resulting collisions (which reconstructs the pre-drift state from a velocity
            // snapshot - see collide.h), then recompute acceleration fresh at the new
            // positions for the next substep's (or next frame's) kick.
            auto t0 = std::chrono::steady_clock::now();
            kickAll(sys, dt);
            driftAll(sys, dt);
            auto t1 = std::chrono::steady_clock::now();
            CollideTimings collideTimings;
            collideParticles(sys, collisionTree, candidates, constants::GRAVITATIONAL_CONSTANT,
                              constants::GRAVITY_SOFTENING_FACTOR, constants::COLLISION_RESTITUTION,
                              constants::COLLISION_SURFACE_GAP, constants::COLLISION_MAX_THREADS, dt, &collideTimings);
            auto t2 = std::chrono::steady_clock::now();
            accumCollideBuildMs += collideTimings.buildMs;
            accumCollideSearchMs += collideTimings.searchMs;
            accumCollideResolveMs += collideTimings.resolveMs;
            for (int32_t i = 0; i < sys.count; i++) {
                totalCandidatesThisFrame += candidates.counts[i];
                densityScratch[i] = std::min((float)candidates.counts[i] / constants::DENSITY_BLUR_THRESHOLD, 1.f);
            }
            if (usePM) {
                computeGravityPM(sys, pmGrid, p3mTree, pmShortRangeTable, constants::PM_GRAVITATIONAL_CONSTANT,
                                  constants::PM_PAIR_SOFTENING_FACTOR, pmWeightsScratch, nullptr, constants::GRAVITY_MAX_THREADS);
                // No position wraparound: particles that drift past a window edge just keep
                // going instead of reappearing on the opposite side. The mesh solve handles
                // them fine - depositCIC/interpolateForceCIC wrap grid indices onto the
                // zero-padded (2x) FFT grid, so a particle up to a full domain-width outside
                // the window still lands in pad cells at its true offset and stays correctly
                // coupled under the isolated-boundary kernel (see pm_gravity.h's header);
                // only particles beyond that alias back around.
            } else {
                computeGravity(sys, gravityTree, constants::GRAVITATIONAL_CONSTANT, constants::GRAVITY_SOFTENING_FACTOR,
                               constants::BARNES_HUT_THETA, constants::QUADTREE_MAX_DEPTH, constants::QUADTREE_LEAF_CAPACITY,
                               constants::GRAVITY_MAX_THREADS);
            }
            auto t3 = std::chrono::steady_clock::now();

            accumKickDriftMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            accumCollideMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
            accumGravityMs += std::chrono::duration<double, std::milli>(t3 - t2).count();
        }
        accumTotalCandidates += totalCandidatesThisFrame;

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        auto tRenderStart = std::chrono::steady_clock::now();
        if (useGpu) {
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            renderer.renderFromBuffers(gpuSim.posBuf, gpuSim.densityOutBuf, sys.count, fbWidth, fbHeight,
                                        (float)windowWidth, (float)windowHeight, cam.zoom, cam.offsetX, cam.offsetY);
        } else {
            renderer.render(sys, fbWidth, fbHeight, (float)windowWidth, (float)windowHeight, densityScratch.data(), cam.zoom, cam.offsetX, cam.offsetY);
        }
        auto tRenderEnd = std::chrono::steady_clock::now();
        accumRenderMs += std::chrono::duration<double, std::milli>(tRenderEnd - tRenderStart).count();

        // Debug panel, bottom-left - mirrors Particles.vue's on-screen readout (fps,
        // particle count, thread ceiling, center of mass), minus the energy section (that
        // needs a pairwise/tree potential-energy pass this port doesn't compute - out of
        // scope for a physics-vs-textures-and-effects benchmark). GPU mode only refreshes
        // COM once per second (in the stats block below) - a per-frame position readback
        // would stall the very pipeline the mode exists to avoid stalling.
        float comX = displayComX, comY = displayComY;
        if (!useGpu) computeCenterOfMass(sys, comX, comY);

        char lineFps[64], lineParticles[64], lineGravity[64], lineThreads[64], lineSubsteps[64], lineZoom[64], lineFrame[64], lineCom[64];
        snprintf(lineFps, sizeof(lineFps), "FPS: %.1f", displayAvgMs > 0.0 ? 1000.0 / displayAvgMs : 0.0);
        snprintf(lineParticles, sizeof(lineParticles), "PARTICLES: %d", sys.count);
        snprintf(lineGravity, sizeof(lineGravity), "GRAVITY: %s", gravityLabel);
        snprintf(lineThreads, sizeof(lineThreads), "THREADS: %d", displayThreadCount);
        snprintf(lineSubsteps, sizeof(lineSubsteps), "SUBSTEPS: %d", displaySubsteps);
        snprintf(lineZoom, sizeof(lineZoom), "ZOOM: %.2fX", cam.zoom);
        snprintf(lineFrame, sizeof(lineFrame), "FRAME: %.2f MS", displayAvgMs);
        snprintf(lineCom, sizeof(lineCom), "COM X: %.2f Y: %.2f", comX, comY);
        const char* lines[] = {"NEBULA NATIVE", lineFps, lineParticles, lineGravity, lineThreads, lineSubsteps, lineZoom, lineFrame, lineCom};
        const int lineCount = 9;

        const float pixelSize = 2.f;
        const float lineHeight = 7.f * pixelSize + 4.f;
        const float padding = 10.f;
        float panelWidth = padding * 2.f;
        for (int i = 0; i < lineCount; i++) {
            panelWidth = std::max(panelWidth, UiRenderer::textWidth(lines[i], pixelSize) + padding * 2.f);
        }
        float panelHeight = padding * 2.f + lineCount * lineHeight;
        float panelX = 20.f;
        float panelY = (float)windowHeight - panelHeight - 20.f;

        ui.begin();
        ui.addQuad(panelX, panelY, panelWidth, panelHeight, 0.04f, 0.03f, 0.09f, 0.55f);
        for (int i = 0; i < lineCount; i++) {
            ui.addText(panelX + padding, panelY + padding + i * lineHeight, pixelSize, lines[i], 0.91f, 0.89f, 1.f, 1.f);
        }
        ui.flush(windowWidth, windowHeight);

        glfwSwapBuffers(window);
        glfwPollEvents();

        //calculate FPS throttle
        auto frameEnd = std::chrono::steady_clock::now();
        double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        
        auto delta = frameTime - frameMs;
        if (useThrottle && delta > 0){
            std::this_thread::sleep_for(std::chrono::milliseconds((int64_t)delta));
        }
        //

        frameEnd = std::chrono::steady_clock::now();
        frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        
        frameTimeAccumMs += frameMs;
        frameCount++;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastFpsPrint).count() >= 1.0) {
            displayAvgMs = frameTimeAccumMs / frameCount;
            if (useGpu) {
                // Once-per-second COM refresh from a full position readback - fine at this
                // cadence, would stall the pipeline if done per frame.
                gpuSim.readPositions(gpuPosReadback);
                double tm = 0.0, cx = 0.0, cy = 0.0;
                for (int32_t i = 0; i < sys.count; i++) {
                    tm += sys.mass[i];
                    cx += (double)sys.mass[i] * gpuPosReadback[(size_t)i * 2];
                    cy += (double)sys.mass[i] * gpuPosReadback[(size_t)i * 2 + 1];
                }
                displayComX = tm > 0 ? (float)(cx / tm) : 0.f;
                displayComY = tm > 0 ? (float)(cy / tm) : 0.f;
                printf("%d particles (GPU): %.2f ms/frame avg (%.1f fps) substeps=%d [gpu physics %.2f, render %.2f] maxAccel=%.4f maxSpeed=%.2f COM=(%.1f,%.1f)\n",
                       sys.count, displayAvgMs, 1000.0 / displayAvgMs, displaySubsteps,
                       accumGravityMs / frameCount, accumRenderMs / frameCount,
                       gpuSim.lastMaxAccel, gpuSim.lastMaxSpeed, displayComX, displayComY);
            } else {
            printf("%d particles: %.2f ms/frame avg (%.1f fps) substeps=%d [kick/drift %.2f, collide %.2f (build %.2f, search %.2f, resolve %.2f), gravity %.2f, render %.2f] avgCandidates/particle=%.1f COM=(%.1f,%.1f)\n",
                   sys.count, displayAvgMs, 1000.0 / displayAvgMs, displaySubsteps,
                   accumKickDriftMs / frameCount, accumCollideMs / frameCount,
                   accumCollideBuildMs / frameCount, accumCollideSearchMs / frameCount, accumCollideResolveMs / frameCount,
                   accumGravityMs / frameCount, accumRenderMs / frameCount,
                   (double)accumTotalCandidates / frameCount / sys.count, comX, comY);
            }
            frameTimeAccumMs = 0.0;
            frameCount = 0;
            accumKickDriftMs = accumCollideMs = accumGravityMs = accumRenderMs = 0.0;
            accumCollideBuildMs = accumCollideSearchMs = accumCollideResolveMs = 0.0;
            accumTotalCandidates = 0;
            lastFpsPrint = now;
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
