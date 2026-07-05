// WebAssembly port of quadtree.ts's Barnes-Hut build + gravity.ts's force traversal - the
// two pieces that together dominate frame time at high particle counts (profiled at ~81%
// of a 50k-particle frame). Ported as a self-contained "compute kernel": the JS side owns
// the real particle data (Particles.vue/simulation.ts's Float32Array SoA system) and
// copies position/mass/radius in, calls compute_gravity(), then copies acceleration back
// out - see gravity.ts's computeGravityWasm for that boundary.
//
// This file's tree (buildTree, below) is also reused for collide.ts's broad-phase nearby-
// particle search (find_nearby_collision_candidates) - the same relationship quadtree.ts
// has with both gravity.ts and merge.ts in the pure-JS code, just carried over to the WASM
// side. That's a deliberate replacement for collide.ts's previous uniform spatial grid
// (spatialGrid.ts, still kept as the JS fallback): a fixed-cell-size grid degrades badly
// under highly non-uniform density (measured directly - the "distributed central mass"
// feature can pack 50+ particles into a single grid cell sized for ~4, since it crams a
// large fraction of the swarm into a tiny fraction of the domain, and that packing gets
// denser over time as gravity keeps accreting more particles into the core). A tree
// doesn't have this failure mode - it just subdivides the dense region further, the same
// way it already does for gravity's own force accuracy.
//
// Tree construction here is a partition-based build, not the one-at-a-time insertion
// quadtree.ts uses: given a range of particle indices, split it into 4 contiguous
// sub-ranges by the node's center (one in-place partition on y, then one each on the
// resulting halves by x), then recurse into each quadrant - stopping once a range is small
// enough to keep as a direct-summation leaf (QUADTREE_LEAF_CAPACITY, see constants.ts)
// rather than subdividing all the way to single-particle leaves. This replaces an earlier
// version of this file that inserted particles one at a time (even after pre-sorting them
// into Morton order for cache locality, still O(n) individual root-to-leaf descents) -
// partitioning gets the same "spatially close particles end up close together in memory"
// property directly, as an O(n log n) sequence of linear in-place partition passes over
// contiguous ranges, with no per-particle pointer-chasing descent at all.
//
// No malloc/std::vector/dynamic containers anywhere in the hot path: every array here is a
// fixed-capacity global, sized once at compile time (MAX_PARTICLES/MAX_NODES below) and
// reused across every call, mirroring quadtree.ts's own reuse-a-persistent-store pattern -
// just without JS's ability to grow a typed array on demand, since WASM linear memory
// growth would invalidate any cached views the JS side holds into it (see gravity.ts). If
// a run ever exceeds MAX_PARTICLES, compute_gravity is a deliberate no-op and the caller
// falls back to the JS implementation - see the capacity check near compute_gravity.
#include <emscripten/emscripten.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

// Comfortably above the largest TOTAL_PARTICLES_OPTIONS entry in constants.ts (50000) so
// normal use never falls back to JS. A generous worst-case node budget: with leaf-capacity
// subdivision stopping early (unlike single-occupant-leaf trees), real trees need
// noticeably fewer nodes than this in practice - the padding is for pathological, highly
// clustered inputs that need many cascading subdivisions before a range shrinks below the
// leaf capacity.
#define MAX_PARTICLES 70000
#define MAX_NODES (MAX_PARTICLES * 4 + 4096)
// Cap on how many candidate indices one find_nearby_collision_candidates() call can return
// - collide.ts's own swept-collision math still exactly filters these afterward (this is a
// broad-phase superset, same contract as quadtree.ts's findNearbyParticles/spatialGrid.ts's
// findNearbyInGrid), so a capped result degrades gracefully: missing a rare, extreme-
// density collision only means it resolves a frame later once the pair's still overlapping
// on the next call, not a correctness failure.
#define MAX_CANDIDATES 4096

// --- Particle data: the JS/WASM boundary ---------------------------------------------
// Input (JS writes these every frame before calling compute_gravity) and output (JS reads
// acc* back out after). treeX/treeY are internal scratch - the clamped insertion
// coordinates, exactly like particleSystem.ts's own treeX/treeY fields - but since nothing
// outside this module ever needs them, they don't need a JS-visible pointer getter.
static float g_posX[MAX_PARTICLES];
static float g_posY[MAX_PARTICLES];
static float g_mass[MAX_PARTICLES];
static float g_radius[MAX_PARTICLES];
static float g_accX[MAX_PARTICLES];
static float g_accY[MAX_PARTICLES];
static float g_treeX[MAX_PARTICLES];
static float g_treeY[MAX_PARTICLES];

// Output buffer for find_nearby_collision_candidates - a single shared buffer, overwritten
// by each call, since collide.ts's JS loop calls this once per particle and always reads
// the result before making the next call (never concurrently, single-threaded control flow
// from JS). Same reuse pattern as g_accX/g_accY: one persistent buffer, no per-call
// allocation.
static int32_t g_candidateBuffer[MAX_CANDIDATES];

// order[] holds particle indices, reordered in place during the build so that every node's
// members occupy a contiguous range [leafStart, leafStart+leafCount) - see partitionRange
// and buildTree. The original g_posX/g_posY/... arrays are never reordered themselves (JS
// reads g_accX/g_accY back by the particle's real, stable index), so this indirection is
// what lets the tree get contiguous-range membership without disturbing that indexing.
static int32_t order[MAX_PARTICLES];

// --- Quadtree node store ---------------------------------------------------------------
// Same flat, parallel-array, next-pointer-traversal design as quadtree.ts's store - see
// that file's header comment for the full rationale (cache-friendly contiguous storage,
// O(1) "skip this subtree" via next instead of recursion). Every leaf - whether it holds
// zero, one, or up to QUADTREE_LEAF_CAPACITY particles - is represented the same way, as a
// range into order[]; there's no separate "single occupant" vs "overflow bucket" case the
// way quadtree.ts (and an earlier version of this file) needed, since partition-based
// construction never needs to cascade-subdivide down to singletons in the first place.
static float nodeX[MAX_NODES];
static float nodeY[MAX_NODES];
static float nodeSize[MAX_NODES];
static int32_t nodeChildren[MAX_NODES]; // index of first of 4 contiguous children, or -1 (leaf)
static int32_t nodeNext[MAX_NODES];     // traversal escape pointer, -1 = nothing more to visit
static int32_t nodeLeafStart[MAX_NODES]; // leaf only: start offset into order[]
static int32_t nodeLeafCount[MAX_NODES]; // leaf only: number of members (0 = empty)
static float nodeMass[MAX_NODES];
static float nodeComX[MAX_NODES];
static float nodeComY[MAX_NODES];
static int32_t parentsStack[MAX_NODES]; // subdivided nodes in subdivision order, for propagate()'s bottom-up pass

static int32_t nodeCount = 0;
static int32_t parentCount = 0;
static int32_t g_quadtreeMaxDepth = 18;
static int32_t g_leafCapacity = 8;

static inline void initNode(int32_t idx, float x, float y, float size, int32_t next) {
    nodeX[idx] = x;
    nodeY[idx] = y;
    nodeSize[idx] = size;
    nodeChildren[idx] = -1;
    nodeNext[idx] = next;
    nodeLeafStart[idx] = 0;
    nodeLeafCount[idx] = 0;
    nodeMass[idx] = 0.f;
    nodeComX[idx] = 0.f;
    nodeComY[idx] = 0.f;
}

// Finalizes a node as a leaf holding order[start, start+count) - computes its aggregate
// mass/COM directly from that range in one pass. Accumulated in double despite the float32
// storage, matching the precision JS gets for free from every Float32Array read being
// promoted to a double before arithmetic happens (see propagate() for the same reasoning
// applied to the bottom-up branch case).
static inline void finalizeLeaf(int32_t node, int32_t start, int32_t count) {
    nodeLeafStart[node] = start;
    nodeLeafCount[node] = count;

    double totalMass = 0.0, comX = 0.0, comY = 0.0;
    for (int32_t k = 0; k < count; k++) {
        int32_t idx = order[start + k];
        double m = g_mass[idx];
        totalMass += m;
        comX += m * g_posX[idx];
        comY += m * g_posY[idx];
    }
    nodeMass[node] = (float)totalMass;
    if (totalMass > 0) {
        nodeComX[node] = (float)(comX / totalMass);
        nodeComY[node] = (float)(comY / totalMass);
    }
}

// In-place partition of order[start, end) by whether the particle's clamped coordinate on
// the given axis falls below `threshold` - the same two-pointer swap-based partition
// std::partition uses, written out directly rather than pulled in via <algorithm> since
// this is the only place that needs it. Returns the split point: [start, split) satisfies
// the predicate, [split, end) doesn't. Uses treeX/treeY (clamped), exactly like
// quadtree.ts's findQuadrant - keeps one far-flung outlier from skewing where the split
// lands relative to the rest of the (tightly packed) swarm.
static inline int32_t partitionRange(int32_t start, int32_t end, bool byY, float threshold) {
    int32_t i = start;
    int32_t j = end;
    while (i < j) {
        int32_t idx = order[i];
        float coord = byY ? g_treeY[idx] : g_treeX[idx];
        if (coord < threshold) {
            i++;
        } else {
            j--;
            int32_t tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }
    return i;
}

// Returns false if the fixed node budget is exhausted instead of writing past the end of
// the node arrays. In practice this should be unreachable at MAX_NODES's generous sizing;
// it exists because C++ has no bounds-checked dynamic growth the way quadtree.ts's
// ensureCapacity does, so silently corrupting memory isn't an acceptable failure mode here
// the way "just allocate more" is in JS.
static bool subdivide(int32_t node, int32_t* outChildren) {
    if (nodeCount + 4 > MAX_NODES) {
        return false;
    }
    int32_t children = nodeCount;
    nodeChildren[node] = children;
    parentsStack[parentCount++] = node;

    float half = nodeSize[node] / 2.f;
    float x = nodeX[node];
    float y = nodeY[node];
    int32_t parentNext = nodeNext[node];

    // NW, NE, SW, SE - matching the south*2+east quadrant convention buildTree's
    // partitioning uses (partition on y first: [start,midY)=north, [midY,end)=south; then
    // each half on x: west first, then east).
    initNode(children + 0, x, y, half, children + 1);
    initNode(children + 1, x + half, y, half, children + 2);
    initNode(children + 2, x, y + half, half, children + 3);
    initNode(children + 3, x + half, y + half, half, parentNext);

    nodeCount += 4;
    *outChildren = children;
    return true;
}

// Explicit work stack for the build below - a range of order[] still waiting to be turned
// into either a leaf or 4 subdivided children. Kept as a fixed array and processed
// depth-first (matching the rest of this codebase's iterative, no-recursion convention)
// rather than actual C++ recursion; sized generously like the other node-related arrays,
// though in practice a LIFO stack never holds more than a handful of pending siblings per
// tree level at once.
struct BuildTask {
    int32_t node;
    int32_t start;
    int32_t end;
    int32_t depth;
};
static BuildTask buildStack[MAX_NODES];
static int32_t buildStackTop = 0;

static inline void pushTask(int32_t node, int32_t start, int32_t end, int32_t depth) {
    buildStack[buildStackTop++] = BuildTask{node, start, end, depth};
}

// Bottom-up aggregate pass, identical in structure to quadtree.ts's propagate() - walks
// parentsStack in reverse so every child is finalized before its parent reads it.
static void propagate() {
    for (int32_t p = parentCount - 1; p >= 0; p--) {
        int32_t node = parentsStack[p];
        int32_t c = nodeChildren[node];

        double m0 = nodeMass[c], m1 = nodeMass[c + 1], m2 = nodeMass[c + 2], m3 = nodeMass[c + 3];
        double totalMass = m0 + m1 + m2 + m3;

        nodeMass[node] = (float)totalMass;
        if (totalMass > 0) {
            nodeComX[node] = (float)((nodeComX[c] * m0 + nodeComX[c + 1] * m1 + nodeComX[c + 2] * m2 + nodeComX[c + 3] * m3) / totalMass);
            nodeComY[node] = (float)((nodeComY[c] * m0 + nodeComY[c + 1] * m1 + nodeComY[c + 2] * m2 + nodeComY[c + 3] * m3) / totalMass);
        }
    }
}

// Builds a fresh tree over the first `count` particles - see quadtree.ts's buildQuadtree
// for the full derivation of the RMS-spread root sizing (unchanged from before: still
// protects against one far-flung outlier blowing up the tree's extent, same as ever).
// Sums that JS gets double precision on "for free" (every Float32Array element promotes to
// a double the moment it's read) are explicitly accumulated in double here too, so this
// doesn't quietly lose more precision than the code it's replacing.
static void buildTree(int32_t count) {
    nodeCount = 0;
    parentCount = 0;
    buildStackTop = 0;

    double totalMass = 0.0, comSumX = 0.0, comSumY = 0.0;
    for (int32_t i = 0; i < count; i++) {
        totalMass += g_mass[i];
        comSumX += (double)g_mass[i] * g_posX[i];
        comSumY += (double)g_mass[i] * g_posY[i];
    }
    float cx = totalMass > 0 ? (float)(comSumX / totalMass) : 0.f;
    float cy = totalMass > 0 ? (float)(comSumY / totalMass) : 0.f;

    double sumSqDeviation = 0.0;
    for (int32_t i = 0; i < count; i++) {
        double dx = (double)g_posX[i] - cx;
        double dy = (double)g_posY[i] - cy;
        sumSqDeviation += dx * dx + dy * dy;
    }
    float rmsSpread = count > 0 ? (float)sqrt(sumSqDeviation / count) : 1.f;

    float halfSize = fmaxf(rmsSpread * 6.f, 10.f);
    float size = halfSize * 2.f;
    float rootX = cx - halfSize;
    float rootY = cy - halfSize;
    // "Just inside the far boundary" - quadtree.ts nudges by a fixed 1e-9, which is exact
    // in JS's native float64 but would silently underflow back to 1.0 in float32 (1e-9 is
    // far below float32's ~1.19e-7 relative precision, so "1 - 1e-9" rounds to exactly 1).
    // Scaling the nudge to the tree's own size keeps it meaningfully representable in
    // float32 regardless of how large or small the swarm's spread is.
    float maxCoord = rootX + size - fmaxf(size * 1e-6f, 1e-6f);

    for (int32_t i = 0; i < count; i++) {
        g_treeX[i] = fminf(fmaxf(g_posX[i], rootX), maxCoord);
        g_treeY[i] = fminf(fmaxf(g_posY[i], rootY), maxCoord);
        order[i] = i;
    }

    initNode(0, rootX, rootY, size, -1);
    nodeCount = 1;

    pushTask(0, 0, count, 0);

    while (buildStackTop > 0) {
        BuildTask task = buildStack[--buildStackTop];
        int32_t node = task.node;
        int32_t start = task.start;
        int32_t end = task.end;
        int32_t depth = task.depth;
        int32_t rangeCount = end - start;

        if (rangeCount <= g_leafCapacity || depth >= g_quadtreeMaxDepth) {
            finalizeLeaf(node, start, rangeCount);
            continue;
        }

        float cx2 = nodeX[node] + nodeSize[node] * 0.5f;
        float cy2 = nodeY[node] + nodeSize[node] * 0.5f;
        // Split on y first (north/south), then each half on x (west/east) - four
        // contiguous ranges in south*2+east (NW, NE, SW, SE) order, matching subdivide().
        int32_t midY = partitionRange(start, end, true, cy2);
        int32_t splitNW = partitionRange(start, midY, false, cx2);
        int32_t splitSW = partitionRange(midY, end, false, cx2);

        int32_t children;
        if (!subdivide(node, &children)) {
            // Node budget exhausted (should be unreachable at this file's MAX_NODES sizing)
            // - finalize as an oversized leaf rather than risk writing past the fixed arrays.
            finalizeLeaf(node, start, rangeCount);
            continue;
        }

        pushTask(children + 0, start, splitNW, depth + 1);
        pushTask(children + 1, splitNW, midY, depth + 1);
        pushTask(children + 2, midY, splitSW, depth + 1);
        pushTask(children + 3, splitSW, end, depth + 1);
    }

    propagate();
}

// --- Gravity traversal -------------------------------------------------------------
// Same softened direct/aggregate split as gravity.ts's applyDirectGravity/
// applyAggregateGravity - see that file for the full physics rationale. The opening-angle
// test is checked before the leaf/branch distinction, not after: a leaf can now hold up to
// QUADTREE_LEAF_CAPACITY particles rather than always exactly one, so "is this leaf/branch
// far enough away to approximate as one point mass" has to be answered first regardless of
// which kind of node it is - only once that test fails (too close to approximate) does it
// matter whether the node is a leaf (direct pairwise sum over its members) or a branch
// (descend further). Checking leaf-ness first, unconditionally direct-summing every leaf's
// members regardless of distance, would mean a leaf far enough away to aggregate cheaply
// still pays for up to QUADTREE_LEAF_CAPACITY individual pairwise calculations - the exact
// per-node work a bigger leaf capacity is supposed to trade away in exchange for fewer,
// cheaper subdivisions.
static inline void applyDirectGravity(int32_t i, int32_t j, float dx, float dy, float distSq, float G, float softeningFactor) {
    float combinedRadius = g_radius[i] + g_radius[j];
    float softenedDistSq = distSq + combinedRadius * combinedRadius * softeningFactor;
    float dist = sqrtf(softenedDistSq);
    float scalar = (G * g_mass[j]) / (softenedDistSq * dist);
    g_accX[i] += dx * scalar;
    g_accY[i] += dy * scalar;
}

static inline void applyAggregateGravity(int32_t i, float otherMass, float dx, float dy, float distSq, float G) {
    float r = g_radius[i];
    float softenedDistSq = distSq + r * r;
    float dist = sqrtf(softenedDistSq);
    float scalar = (G * otherMass) / (softenedDistSq * dist);
    g_accX[i] += dx * scalar;
    g_accY[i] += dy * scalar;
}

static void applyTreeGravity(int32_t i, float thetaSq, float G, float softeningFactor) {
    float px = g_posX[i];
    float py = g_posY[i];
    int32_t node = 0;

    while (node != -1) {
        float mass = nodeMass[node];
        if (mass == 0.f) {
            node = nodeNext[node];
            continue;
        }

        float dx = nodeComX[node] - px;
        float dy = nodeComY[node] - py;
        float distSq = dx * dx + dy * dy;

        if (nodeSize[node] * nodeSize[node] < distSq * thetaSq) {
            // Far enough to treat as one point mass, whether this node is a leaf or a
            // branch - see the comment above this function for why the distance check has
            // to come before the leaf/branch distinction now that leaves can hold more
            // than one particle.
            applyAggregateGravity(i, mass, dx, dy, distSq, G);
            node = nodeNext[node];
        } else if (nodeChildren[node] == -1) {
            // Too close to approximate - direct pairwise sum over this leaf's members.
            int32_t start = nodeLeafStart[node];
            int32_t cnt = nodeLeafCount[node];
            for (int32_t k = 0; k < cnt; k++) {
                int32_t j = order[start + k];
                if (j == i) continue;
                float odx = g_posX[j] - px;
                float ody = g_posY[j] - py;
                applyDirectGravity(i, j, odx, ody, odx * odx + ody * ody, G, softeningFactor);
            }
            node = nodeNext[node];
        } else {
            node = nodeChildren[node];
        }
    }
}

// Runs applyTreeGravity for a contiguous slice [startI, endI) of particles - the unit of
// work compute_gravity hands to each thread. Splitting by raw particle index rather than
// anything spatial is deliberate and safe here: particle index is spawn order, which has
// no lasting correlation with where a particle ends up after the swarm's been evolving for
// a while, so a plain contiguous split already spreads spatial density (and therefore
// per-particle traversal cost) evenly across threads on average, without needing an actual
// work-stealing scheduler.
static void computeGravityForRange(int32_t startI, int32_t endI, float thetaSq, float G, float softeningFactor) {
    for (int32_t i = startI; i < endI; i++) {
        applyTreeGravity(i, thetaSq, G, softeningFactor);
    }
}

// --- Collision broad-phase (find_nearby_collision_candidates) -----------------------
// Direct port of quadtree.ts's findNearbyParticles onto this file's partition-based tree:
// collect every particle within searchRadius of particle i, pruning subtrees whose
// bounding box can't possibly contain a point that close (closest-point-on-box test).
// Unlike gravity's traversal, this never aggregates - collision needs exact candidate
// indices, not a force approximation, so every node visit either prunes the whole subtree
// or (for a leaf) adds its members directly.
static int32_t findNearbyCollisionCandidates(int32_t i, float searchRadius) {
    float px = g_posX[i];
    float py = g_posY[i];
    float searchRadiusSq = searchRadius * searchRadius;
    int32_t outCount = 0;
    int32_t node = 0;

    while (node != -1) {
        if (nodeMass[node] == 0.f) {
            node = nodeNext[node]; // empty subtree - nothing to prune-test, just skip
            continue;
        }

        float nodeXv = nodeX[node];
        float nodeYv = nodeY[node];
        float nodeSizeV = nodeSize[node];
        float closestX = fminf(fmaxf(px, nodeXv), nodeXv + nodeSizeV);
        float closestY = fminf(fmaxf(py, nodeYv), nodeYv + nodeSizeV);
        float dx = px - closestX;
        float dy = py - closestY;

        if (dx * dx + dy * dy > searchRadiusSq) {
            node = nodeNext[node]; // whole subtree pruned - its box can't be close enough
            continue;
        }

        if (nodeChildren[node] == -1) {
            int32_t start = nodeLeafStart[node];
            int32_t cnt = nodeLeafCount[node];
            for (int32_t k = 0; k < cnt && outCount < MAX_CANDIDATES; k++) {
                int32_t j = order[start + k];
                if (j != i) {
                    g_candidateBuffer[outCount++] = j;
                }
            }
            node = nodeNext[node];
        } else {
            node = nodeChildren[node]; // box is close enough - check children individually
        }
    }

    return outCount;
}

extern "C" {

// Pointer getters: called once by the JS side right after the module loads, so it can
// compute HEAPF32 offsets for the input/output arrays without any per-frame marshaling
// overhead - see gravity.ts's initGravityWasm. Memory growth is disabled for this module
// (see package.json's build:wasm/build:wasm:threaded scripts), so these addresses - and
// any HEAPF32 view built from them - stay valid for the module's entire lifetime.
EMSCRIPTEN_KEEPALIVE float* get_pos_x_ptr() { return g_posX; }
EMSCRIPTEN_KEEPALIVE float* get_pos_y_ptr() { return g_posY; }
EMSCRIPTEN_KEEPALIVE float* get_mass_ptr() { return g_mass; }
EMSCRIPTEN_KEEPALIVE float* get_radius_ptr() { return g_radius; }
EMSCRIPTEN_KEEPALIVE float* get_acc_x_ptr() { return g_accX; }
EMSCRIPTEN_KEEPALIVE float* get_acc_y_ptr() { return g_accY; }
EMSCRIPTEN_KEEPALIVE int32_t* get_candidate_buffer_ptr() { return g_candidateBuffer; }
EMSCRIPTEN_KEEPALIVE int get_max_particles() { return MAX_PARTICLES; }
EMSCRIPTEN_KEEPALIVE int get_max_candidates() { return MAX_CANDIDATES; }

// Builds a fresh tree from whatever's currently sitting in g_posX/g_posY/g_mass/g_radius
// (JS writes those via the pointers above before calling this) and fills g_accX/g_accY
// with the pure gravitational contribution - not += onto whatever was there before, since
// this module has no notion of the constant external GRAVITY field particleSystem.ts's
// resetAccelerationAll seeds acceleration with; the caller adds this output onto its own
// acceleration array instead (see gravity.ts). Silently does nothing if count exceeds this
// module's fixed capacity - the caller is expected to check get_max_particles() first and
// fall back to the JS implementation rather than ever hitting this.
EMSCRIPTEN_KEEPALIVE void compute_gravity(int count, float G, float softeningFactor, float theta, int quadtreeMaxDepth, int leafCapacity, int maxThreads) {
    if (count < 0 || count > MAX_PARTICLES) {
        return;
    }
    g_quadtreeMaxDepth = quadtreeMaxDepth;
    g_leafCapacity = leafCapacity < 1 ? 1 : leafCapacity;
    buildTree(count);

    float thetaSq = theta * theta;
    for (int i = 0; i < count; i++) {
        g_accX[i] = 0.f;
        g_accY[i] = 0.f;
    }
    // Each particle's traversal only ever reads the (already fully built) tree and writes
    // its own g_accX[i]/g_accY[i] - never any other particle's, and never any shared
    // mutable state - so splitting this loop across threads is race-free by construction,
    // the same "embarrassingly parallel" property the reference Rust implementation's
    // par_iter_mut exploits. Tree *construction* stays single-threaded (see buildTree) -
    // it's already the cheaper half of this function per profiling, and parallelizing the
    // partition-based build safely needs its own work-distribution scheme, not just a loop
    // split. Plain std::thread rather than OpenMP: this Emscripten install doesn't ship
    // libomp/omp.h, but -pthread alone is enough for std::thread, which is all this needs.
    //
    // hardware_concurrency() reflects navigator.hardwareConcurrency once compiled with
    // -pthread (see package.json's build:wasm:threaded script) - on a non-threaded build
    // (package.json's plain build:wasm, no -pthread) it reports 0 (per the standard's
    // documented fallback for "not computable"), which the clamp below turns into a single
    // sequential pass. That's what lets gravity.cpp itself stay identical between the two
    // builds - whether this actually runs in parallel is entirely a compile-flag / runtime
    // (crossOriginIsolated) question, never a code-path branch here.
    //
    // maxThreads (constants.ts's GRAVITY_MAX_THREADS) is a user-tunable ceiling on top of
    // hardware_concurrency(), not a replacement for it - std::min below still never spawns
    // more threads than the browser reports logical cores for. Requesting more threads than
    // there are cores doesn't add throughput (there's no more CPU to schedule them onto),
    // it only adds cost: every thread here is spawned fresh and joined every single
    // compute_gravity call - i.e. up to 60 times a second - so oversubscribing relative to
    // real hardware just means paying that spawn/join overhead for threads that end up
    // time-slicing against each other on the same cores, doing the same total work slower
    // than fewer, fully-scheduled threads would have.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    int32_t threadCap = maxThreads < 1 ? 1 : maxThreads;
    int32_t threadCount = hwThreads < 1 ? 1 : (int32_t)std::min(hwThreads, (unsigned int)threadCap);
    if (threadCount > count) {
        threadCount = count > 0 ? count : 1;
    }

    if (threadCount <= 1) {
        computeGravityForRange(0, count, thetaSq, G, softeningFactor);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        int32_t chunk = (count + threadCount - 1) / threadCount;
        for (int32_t t = 0; t < threadCount; t++) {
            int32_t startI = t * chunk;
            int32_t endI = std::min(startI + chunk, count);
            if (startI >= endI) break;
            workers.emplace_back(computeGravityForRange, startI, endI, thetaSq, G, softeningFactor);
        }
        for (auto& w : workers) {
            w.join();
        }
    }
}

// Builds a fresh tree for collide.ts's broad-phase search, from whatever's currently
// sitting in g_posX/g_posY/g_mass/g_radius (same input buffers compute_gravity uses -
// collision and gravity never run concurrently within a frame, so sharing them is safe;
// see simulation.ts's stepSimulation for why collision always runs first and gravity
// always rebuilds fresh afterward regardless). Takes its own leafCapacity separate from
// compute_gravity's: gravity wants coarser leaves (batches more direct-sum work per node
// to justify fewer subdivisions), collision wants finer ones (fewer false-positive
// candidates needing a full swept-collision check in JS) - see constants.ts's
// COLLISION_TREE_LEAF_CAPACITY. Silently does nothing if count exceeds this module's fixed
// capacity, same contract as compute_gravity - the caller checks get_max_particles() first.
EMSCRIPTEN_KEEPALIVE void build_collision_tree(int count, int quadtreeMaxDepth, int leafCapacity) {
    if (count < 0 || count > MAX_PARTICLES) {
        return;
    }
    g_quadtreeMaxDepth = quadtreeMaxDepth;
    g_leafCapacity = leafCapacity < 1 ? 1 : leafCapacity;
    buildTree(count);
}

// Temporary diagnostic: largest leaf occupancy in the tree just built. Not used by
// collide.ts - only for tuning QUADTREE_MAX_DEPTH/leaf capacity against real density.
EMSCRIPTEN_KEEPALIVE int debug_max_leaf_occupancy() {
    int32_t maxCount = 0;
    for (int32_t n = 0; n < nodeCount; n++) {
        if (nodeChildren[n] == -1 && nodeLeafCount[n] > maxCount) {
            maxCount = nodeLeafCount[n];
        }
    }
    return maxCount;
}

// Collects every particle within searchRadius of particle i into g_candidateBuffer (read
// back via get_candidate_buffer_ptr) and returns how many were found. Must be called after
// build_collision_tree, once per particle - see collide.ts's findNearbyWasm for the JS side
// of this loop, and findNearbyCollisionCandidates above for the traversal itself.
EMSCRIPTEN_KEEPALIVE int find_nearby_collision_candidates(int i, float searchRadius) {
    return findNearbyCollisionCandidates(i, searchRadius);
}

} // extern "C"
