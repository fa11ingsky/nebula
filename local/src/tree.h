#pragma once
// Direct native port of src/wasm/gravity.cpp's partition-based Barnes-Hut tree - see that
// file's header comment for the full rationale (contiguous-range partitioning instead of
// one-at-a-time insertion, why leaves hold up to leafCapacity particles instead of always
// one, next-pointer iterative traversal instead of recursion). The only real change here is
// dropping the WASM/JS heap boundary entirely: this reads straight from a ParticleSystem's
// own arrays instead of copying into a separate g_posX/g_posY/... staging buffer, since a
// native app has no marshaling boundary to cross in the first place.
//
// Instantiated twice by the caller (one SpatialTree for gravity, one for collision), exactly
// like gravity.ts keeps two separate WASM trees - they want different leaf capacity/max depth
// tuning (see constants.h) and get rebuilt independently every frame anyway.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "particle_system.h"

struct SpatialTree {
    int32_t maxNodes = 0;
    float* nodeX = nullptr;
    float* nodeY = nullptr;
    float* nodeSize = nullptr;
    int32_t* nodeChildren = nullptr; // index of first of 4 contiguous children, or -1 (leaf)
    int32_t* nodeNext = nullptr;     // traversal escape pointer, -1 = nothing more to visit
    int32_t* nodeLeafStart = nullptr;
    int32_t* nodeLeafCount = nullptr;
    float* nodeMass = nullptr;
    float* nodeComX = nullptr;
    float* nodeComY = nullptr;
    int32_t* parentsStack = nullptr;
    int32_t* order = nullptr;   // particle indices, reordered so each node's members are contiguous
    float* treeX = nullptr;     // clamped insertion coordinates (see buildTree)
    float* treeY = nullptr;

    int32_t nodeCount = 0;
    int32_t parentCount = 0;
    int32_t leafCapacity = 8;
    int32_t maxDepth = 18;

    struct BuildTask {
        int32_t node, start, end, depth;
    };
    BuildTask* buildStack = nullptr;
    int32_t buildStackTop = 0;

    void allocate(int32_t particleCapacity) {
        maxNodes = particleCapacity * 4 + 4096;
        nodeX = new float[maxNodes];
        nodeY = new float[maxNodes];
        nodeSize = new float[maxNodes];
        nodeChildren = new int32_t[maxNodes];
        nodeNext = new int32_t[maxNodes];
        nodeLeafStart = new int32_t[maxNodes];
        nodeLeafCount = new int32_t[maxNodes];
        nodeMass = new float[maxNodes];
        nodeComX = new float[maxNodes];
        nodeComY = new float[maxNodes];
        parentsStack = new int32_t[maxNodes];
        order = new int32_t[particleCapacity];
        treeX = new float[particleCapacity];
        treeY = new float[particleCapacity];
        buildStack = new BuildTask[maxNodes];
    }

    inline void initNode(int32_t idx, float x, float y, float size, int32_t next) {
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

    inline void finalizeLeaf(const ParticleSystem& sys, int32_t node, int32_t start, int32_t count) {
        nodeLeafStart[node] = start;
        nodeLeafCount[node] = count;
        double totalMass = 0.0, comX = 0.0, comY = 0.0;
        for (int32_t k = 0; k < count; k++) {
            int32_t idx = order[start + k];
            double m = sys.mass[idx];
            totalMass += m;
            comX += m * sys.posX[idx];
            comY += m * sys.posY[idx];
        }
        nodeMass[node] = (float)totalMass;
        if (totalMass > 0) {
            nodeComX[node] = (float)(comX / totalMass);
            nodeComY[node] = (float)(comY / totalMass);
        }
    }

    inline int32_t partitionRange(int32_t start, int32_t end, bool byY, float threshold) {
        int32_t i = start, j = end;
        while (i < j) {
            int32_t idx = order[i];
            float coord = byY ? treeY[idx] : treeX[idx];
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

    inline bool subdivide(int32_t node, int32_t* outChildren) {
        if (nodeCount + 4 > maxNodes) return false;
        int32_t children = nodeCount;
        nodeChildren[node] = children;
        parentsStack[parentCount++] = node;

        float half = nodeSize[node] / 2.f;
        float x = nodeX[node];
        float y = nodeY[node];
        int32_t parentNext = nodeNext[node];

        initNode(children + 0, x, y, half, children + 1);
        initNode(children + 1, x + half, y, half, children + 2);
        initNode(children + 2, x, y + half, half, children + 3);
        initNode(children + 3, x + half, y + half, half, parentNext);

        nodeCount += 4;
        *outChildren = children;
        return true;
    }

    inline void pushTask(int32_t node, int32_t start, int32_t end, int32_t depth) {
        buildStack[buildStackTop++] = BuildTask{node, start, end, depth};
    }

    void propagate() {
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

    void build(const ParticleSystem& sys, int32_t count) {
        nodeCount = 0;
        parentCount = 0;
        buildStackTop = 0;

        double totalMass = 0.0, comSumX = 0.0, comSumY = 0.0;
        for (int32_t i = 0; i < count; i++) {
            totalMass += sys.mass[i];
            comSumX += (double)sys.mass[i] * sys.posX[i];
            comSumY += (double)sys.mass[i] * sys.posY[i];
        }
        float cx = totalMass > 0 ? (float)(comSumX / totalMass) : 0.f;
        float cy = totalMass > 0 ? (float)(comSumY / totalMass) : 0.f;

        double sumSqDeviation = 0.0;
        for (int32_t i = 0; i < count; i++) {
            double dx = (double)sys.posX[i] - cx;
            double dy = (double)sys.posY[i] - cy;
            sumSqDeviation += dx * dx + dy * dy;
        }
        float rmsSpread = count > 0 ? (float)sqrt(sumSqDeviation / count) : 1.f;

        float halfSize = fmaxf(rmsSpread * 6.f, 10.f);
        float size = halfSize * 2.f;
        float rootX = cx - halfSize;
        float rootY = cy - halfSize;
        float maxCoord = rootX + size - fmaxf(size * 1e-6f, 1e-6f);

        for (int32_t i = 0; i < count; i++) {
            treeX[i] = fminf(fmaxf(sys.posX[i], rootX), maxCoord);
            treeY[i] = fminf(fmaxf(sys.posY[i], rootY), maxCoord);
            order[i] = i;
        }

        initNode(0, rootX, rootY, size, -1);
        nodeCount = 1;
        pushTask(0, 0, count, 0);

        while (buildStackTop > 0) {
            BuildTask task = buildStack[--buildStackTop];
            int32_t node = task.node, start = task.start, end = task.end, depth = task.depth;
            int32_t rangeCount = end - start;

            if (rangeCount <= leafCapacity || depth >= maxDepth) {
                finalizeLeaf(sys, node, start, rangeCount);
                continue;
            }

            float cx2 = nodeX[node] + nodeSize[node] * 0.5f;
            float cy2 = nodeY[node] + nodeSize[node] * 0.5f;
            int32_t midY = partitionRange(start, end, true, cy2);
            int32_t splitNW = partitionRange(start, midY, false, cx2);
            int32_t splitSW = partitionRange(midY, end, false, cx2);

            int32_t children;
            if (!subdivide(node, &children)) {
                finalizeLeaf(sys, node, start, rangeCount);
                continue;
            }

            pushTask(children + 0, start, splitNW, depth + 1);
            pushTask(children + 1, splitNW, midY, depth + 1);
            pushTask(children + 2, midY, splitSW, depth + 1);
            pushTask(children + 3, splitSW, end, depth + 1);
        }

        propagate();
    }

    // Collects every particle within searchRadius of particle i into outBuffer - broad-phase
    // superset, same contract as collide.cpp's exact swept-collision filtering afterward.
    inline int32_t findNearbyInto(const ParticleSystem& sys, int32_t i, float searchRadius, int32_t* outBuffer, int32_t maxOut) const {
        float px = sys.posX[i];
        float py = sys.posY[i];
        float searchRadiusSq = searchRadius * searchRadius;
        int32_t outCount = 0;
        int32_t node = 0;

        while (node != -1) {
            if (nodeMass[node] == 0.f) {
                node = nodeNext[node];
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
                node = nodeNext[node];
                continue;
            }

            if (nodeChildren[node] == -1) {
                int32_t start = nodeLeafStart[node];
                int32_t cnt = nodeLeafCount[node];
                for (int32_t k = 0; k < cnt && outCount < maxOut; k++) {
                    int32_t j = order[start + k];
                    if (j != i) outBuffer[outCount++] = j;
                }
                node = nodeNext[node];
            } else {
                node = nodeChildren[node];
            }
        }

        return outCount;
    }
};
