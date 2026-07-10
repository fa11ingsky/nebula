#pragma once
// Native port of src/lib/sim/gravity.ts's applyTreeGravity/computeGravity, operating on
// tree.h's SpatialTree instead of gravity.cpp's WASM-boundary staging buffers - see that
// file for the softened direct/aggregate physics rationale (unchanged here).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include "particle_system.h"
#include "tree.h"
#include "constants.h"

inline void applyDirectGravity(ParticleSystem& sys, int32_t i, int32_t j, float dx, float dy, float distSq, float G, float softeningFactor) {
    float combinedRadius = sys.radius[i] + sys.radius[j];
    // Softening length is radius-proportional but capped (see constants.h's
    // GRAVITY_SOFTENING_MAX_LENGTH) - without the cap, heavy bodies (few-particle runs,
    // the central mass) get a softening zone bigger than their whole orbit and stop
    // gravitating at all. collide.h's softenedPairPE must apply the identical cap - the
    // overlap-resolution energy budget has to price PE in the same field this force
    // actually exerts.
    float softeningSq = std::min(combinedRadius * combinedRadius * softeningFactor,
                                  constants::GRAVITY_SOFTENING_MAX_LENGTH * constants::GRAVITY_SOFTENING_MAX_LENGTH);
    float softenedDistSq = distSq + softeningSq;
    float dist = sqrtf(softenedDistSq);
    float scalar = (G * sys.mass[j]) / (softenedDistSq * dist);
    sys.accX[i] += dx * scalar;
    sys.accY[i] += dy * scalar;
}

inline void applyAggregateGravity(ParticleSystem& sys, int32_t i, float otherMass, float dx, float dy, float distSq, float G) {
    float r = sys.radius[i];
    float softenedDistSq = distSq + r * r;
    float dist = sqrtf(softenedDistSq);
    float scalar = (G * otherMass) / (softenedDistSq * dist);
    sys.accX[i] += dx * scalar;
    sys.accY[i] += dy * scalar;
}

inline void applyTreeGravity(ParticleSystem& sys, int32_t i, const SpatialTree& tree, float thetaSq, float G, float softeningFactor) {
    float px = sys.posX[i];
    float py = sys.posY[i];
    int32_t node = 0;

    while (node != -1) {
        float mass = tree.nodeMass[node];
        if (mass == 0.f) {
            node = tree.nodeNext[node];
            continue;
        }

        float dx = tree.nodeComX[node] - px;
        float dy = tree.nodeComY[node] - py;
        float distSq = dx * dx + dy * dy;

        if (tree.nodeSize[node] * tree.nodeSize[node] < distSq * thetaSq) {
            applyAggregateGravity(sys, i, mass, dx, dy, distSq, G);
            node = tree.nodeNext[node];
        } else if (tree.nodeChildren[node] == -1) {
            int32_t start = tree.nodeLeafStart[node];
            int32_t cnt = tree.nodeLeafCount[node];
            for (int32_t k = 0; k < cnt; k++) {
                int32_t j = tree.order[start + k];
                if (j == i) continue;
                float odx = sys.posX[j] - px;
                float ody = sys.posY[j] - py;
                applyDirectGravity(sys, i, j, odx, ody, odx * odx + ody * ody, G, softeningFactor);
            }
            node = tree.nodeNext[node];
        } else {
            node = tree.nodeChildren[node];
        }
    }
}

inline void computeGravityForRange(ParticleSystem& sys, const SpatialTree& tree, int32_t startI, int32_t endI, float thetaSq, float G, float softeningFactor) {
    for (int32_t i = startI; i < endI; i++) {
        applyTreeGravity(sys, i, tree, thetaSq, G, softeningFactor);
    }
}

// Builds a fresh gravity tree and applies its force to every particle, splitting the
// per-particle traversal across up to maxThreads real OS threads - see gravity.cpp's
// compute_gravity for why this is race-free (each thread only ever writes its own
// accX[i]/accY[i], and only ever reads the already-built, never-mutated tree). Unlike the
// WASM build, hardware_concurrency() here reflects the actual machine, not a
// crossOriginIsolated browser tab - this is the whole point of the native benchmark.
inline void computeGravity(ParticleSystem& sys, SpatialTree& tree, float G, float softeningFactor, float theta, int32_t quadtreeMaxDepth, int32_t leafCapacity, int32_t maxThreads) {
    int32_t count = sys.count;
    tree.maxDepth = quadtreeMaxDepth;
    tree.leafCapacity = leafCapacity < 1 ? 1 : leafCapacity;
    tree.build(sys, count);

    float thetaSq = theta * theta;
    for (int32_t i = 0; i < count; i++) {
        sys.accX[i] = 0.f;
        sys.accY[i] = 0.f;
    }

    unsigned int hwThreads = std::thread::hardware_concurrency();
    int32_t threadCap = maxThreads < 1 ? 1 : maxThreads;
    int32_t threadCount = hwThreads < 1 ? 1 : (int32_t)std::min(hwThreads, (unsigned int)threadCap);
    if (threadCount > count) threadCount = count > 0 ? count : 1;

    if (threadCount <= 1) {
        computeGravityForRange(sys, tree, 0, count, thetaSq, G, softeningFactor);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        int32_t chunk = (count + threadCount - 1) / threadCount;
        for (int32_t t = 0; t < threadCount; t++) {
            int32_t startI = t * chunk;
            int32_t endI = std::min(startI + chunk, count);
            if (startI >= endI) break;
            workers.emplace_back(computeGravityForRange, std::ref(sys), std::ref(tree), startI, endI, thetaSq, G, softeningFactor);
        }
        for (auto& w : workers) w.join();
    }
}
