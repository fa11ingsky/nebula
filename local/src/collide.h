#pragma once
// Native port of src/lib/sim/collide.ts, scoped to this port's non-merging mode only:
// touching particles always bounce (resolveImpulse) or get pushed apart without
// overlapping (resolveOverlap) - merge.ts's fuse-into-one-body path was never ported, so
// there's no merge branch to choose between here, unlike the web app.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include "particle_system.h"
#include "tree.h"
#include "constants.h"

#define MAX_CANDIDATES_PER_PARTICLE 128

// Flat, open-addressing "seen this frame" set for resolvedPairs - allocated once and
// reused every frame via a generation counter (clearing is O(1): just bump currentGen,
// never touch the arrays), instead of std::unordered_set, which both re-allocates its own
// table AND heap-allocates a node per insert, every single frame.
//
// Profiling at 50k particles showed collision resolution cost scaling perfectly linearly
// with candidate count (no hidden quadratic blowup), but at ~2.5x the cost it should have
// - traced to collideParticles calling insertIfNew on every single examined candidate, not
// just the ones that actually resolve. Gating the dedup check at the point of an actual
// resolveOverlap/resolveImpulse call instead (see collideParticles) cut real hash-table
// traffic down to just genuine collisions - the vast majority of examined candidates now
// never touch this structure at all, since their own "no resolution" branches are cheap,
// direction-invariant math with nothing to deduplicate.
struct PairSet {
    std::vector<int64_t> keys;
    std::vector<uint32_t> gen;
    uint32_t currentGen = 0;
    size_t mask = 0;

    void allocate(size_t minSlots) {
        size_t tableSize = 1;
        while (tableSize < minSlots) tableSize <<= 1;
        keys.assign(tableSize, 0);
        gen.assign(tableSize, 0);
        mask = tableSize - 1;
        currentGen = 0;
    }

    inline void beginFrame() {
        currentGen++;
        if (currentGen == 0) { // wrapped around (essentially never hit) - fall back to a real clear
            std::fill(gen.begin(), gen.end(), 0);
            currentGen = 1;
        }
    }

    // Returns true if this key wasn't already marked this frame (and marks it); false if
    // it was already seen. Linear probing capped at a small, fixed number of slots rather
    // than the whole table - sizing the table for the theoretical worst case (every single
    // candidate examination a distinct pair) instead of typical load kept load factor safe
    // but blew up cache locality (measured directly: a several-hundred-MB table pushed
    // resolve-stage cost the wrong way). A modest table plus this cap gets the actual
    // win - if a key isn't found within PROBE_LIMIT slots, this just treats it as new and
    // gives up on dedup for that one pair (a rare, bounded double-resolution in only the
    // densest pile-ups, never an unbounded stall) rather than degrading toward O(n) probes
    // per insert as the table fills up.
    static constexpr int32_t PROBE_LIMIT = 24;
    inline bool insertIfNew(int64_t key) {
        uint64_t h = (uint64_t)key * 0x9E3779B97F4A7C15ULL;
        size_t idx = (size_t)(h >> 32) & mask;
        int32_t limit = (int32_t)std::min((size_t)PROBE_LIMIT, mask + 1);
        for (int32_t probes = 0; probes < limit; probes++) {
            if (gen[idx] != currentGen) {
                gen[idx] = currentGen;
                keys[idx] = key;
                return true;
            }
            if (keys[idx] == key) return false;
            idx = (idx + 1) & mask;
        }
        return true;
    }
};

struct CollisionCandidates {
    int32_t* buffer = nullptr;  // capacity * MAX_CANDIDATES_PER_PARTICLE, one fixed slot per particle
    int32_t* counts = nullptr;
    float* origVelX = nullptr;  // velocity snapshot at frame entry - see collide.ts's header comment on why this must be frozen
    float* origVelY = nullptr;
    PairSet resolvedPairs;

    void allocate(int32_t capacity) {
        buffer = new int32_t[(size_t)capacity * MAX_CANDIDATES_PER_PARTICLE];
        counts = new int32_t[capacity];
        origVelX = new float[capacity];
        origVelY = new float[capacity];
        // Sized for typical load (most pairs get discovered from both directions, so
        // distinct pairs/frame is well under capacity*MAX_CANDIDATES_PER_PARTICLE - see
        // PairSet's header comment for why sizing at that theoretical worst case instead
        // was actually a regression) - PROBE_LIMIT is what keeps this safe under a genuine
        // pathological spike, not table size.
        resolvedPairs.allocate((size_t)capacity * 4);
    }
};

// Matches energy.ts's direct-pair PE formula exactly, so resolveOverlap's energy budget is
// the same quantity the rest of the sim would report.
inline float softenedPairPE(const ParticleSystem& sys, int32_t i, int32_t j, float dist, float G, float softeningFactor) {
    float combinedRadius = sys.radius[i] + sys.radius[j];
    // Same capped softening as gravity.h's applyDirectGravity - the two must stay
    // identical or resolveOverlap's energy budget prices PE in a different field than the
    // force the pair actually feels.
    float softeningSq = std::min(combinedRadius * combinedRadius * softeningFactor,
                                  constants::GRAVITY_SOFTENING_MAX_LENGTH * constants::GRAVITY_SOFTENING_MAX_LENGTH);
    float softenedDistSq = dist * dist + softeningSq;
    return -(G * sys.mass[i] * sys.mass[j]) / sqrtf(softenedDistSq);
}

// See collide.ts's resolveOverlap for the full derivation: pays for the position shift by
// removing exactly that much kinetic energy from the pair's relative motion along the
// normal, rather than shoving them apart for free (which would inject energy from nothing).
inline void resolveOverlap(ParticleSystem& sys, int32_t i, int32_t j, float G, float softeningFactor, float gap) {
    float dx = sys.posX[j] - sys.posX[i];
    float dy = sys.posY[j] - sys.posY[i];
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist == 0.f) dist = 1e-6f;
    float touchDistance = sys.radius[i] + sys.radius[j] + gap;

    if (dist >= touchDistance) return;

    float nx = dx / dist;
    float ny = dy / dist;

    float m1 = sys.mass[i];
    float m2 = sys.mass[j];
    float invM1 = sys.fixed[i] ? 0.f : 1.f / m1;
    float invM2 = sys.fixed[j] ? 0.f : 1.f / m2;
    float mu = 1.f / (invM1 + invM2);

    float relVx = sys.velX[i] - sys.velX[j];
    float relVy = sys.velY[i] - sys.velY[j];
    float vn = relVx * nx + relVy * ny;

    float availableEnergy = 0.5f * mu * vn * vn;
    float fullSeparationCost = softenedPairPE(sys, i, j, touchDistance, G, softeningFactor) - softenedPairPE(sys, i, j, dist, G, softeningFactor);

    float targetDist;
    if (availableEnergy >= fullSeparationCost) {
        targetDist = touchDistance;
    } else {
        float combinedRadius = sys.radius[i] + sys.radius[j];
        // Must invert softenedPairPE's exact formula, capped softening included.
        float softeningSq = std::min(combinedRadius * combinedRadius * softeningFactor,
                                      constants::GRAVITY_SOFTENING_MAX_LENGTH * constants::GRAVITY_SOFTENING_MAX_LENGTH);
        float gm1m2 = G * m1 * m2;
        float invSqrtR0 = 1.f / sqrtf(dist * dist + softeningSq);
        float k = invSqrtR0 - availableEnergy / gm1m2;
        if (k <= 0.f) {
            targetDist = touchDistance;
        } else {
            targetDist = sqrtf(std::max(1.f / (k * k) - softeningSq, 0.f));
        }
    }

    if (targetDist <= dist) return;

    float weight1 = invM1 / (invM1 + invM2);
    float weight2 = invM2 / (invM1 + invM2);
    float shift = targetDist - dist;
    sys.posX[i] -= weight1 * shift * nx;
    sys.posY[i] -= weight1 * shift * ny;
    sys.posX[j] += weight2 * shift * nx;
    sys.posY[j] += weight2 * shift * ny;

    float spentEnergy = softenedPairPE(sys, i, j, targetDist, G, softeningFactor) - softenedPairPE(sys, i, j, dist, G, softeningFactor);
    float discriminant = std::max(vn * vn - (2.f * spentEnergy) / mu, 0.f);
    float vnNew = (vn > 0 ? 1.f : (vn < 0 ? -1.f : 0.f)) * sqrtf(discriminant);
    float impulse = mu * (vn - vnNew);

    sys.velX[i] -= impulse * invM1 * nx;
    sys.velY[i] -= impulse * invM1 * ny;
    sys.velX[j] += impulse * invM2 * nx;
    sys.velY[j] += impulse * invM2 * ny;
}

// See collide.ts's resolveImpulse: momentum-conserving bounce along the collision normal,
// with the fixed-particle (infinite mass) limit falling out of invM1/invM2 being exactly 0.
// `dt` is this call's substep length (1.0 for a full, un-substepped frame) - remainingT is a
// dimensionless fraction of THIS substep, so converting it to an actual position correction
// needs multiplying by dt, the same way driftAll's own position update does.
inline void resolveImpulse(ParticleSystem& sys, int32_t i, int32_t j, float nx, float ny, float remainingT, float restitution, float dt) {
    float relVxIJ = sys.velX[i] - sys.velX[j];
    float relVyIJ = sys.velY[i] - sys.velY[j];
    float vn = relVxIJ * nx + relVyIJ * ny;

    float m1 = sys.mass[i];
    float m2 = sys.mass[j];
    float invM1 = sys.fixed[i] ? 0.f : 1.f / m1;
    float invM2 = sys.fixed[j] ? 0.f : 1.f / m2;
    float impulse = ((1.f + restitution) * vn) / (invM1 + invM2);

    float oldVelXi = sys.velX[i], oldVelYi = sys.velY[i];
    float oldVelXj = sys.velX[j], oldVelYj = sys.velY[j];

    float newVelXi = oldVelXi - impulse * invM1 * nx;
    float newVelYi = oldVelYi - impulse * invM1 * ny;
    float newVelXj = oldVelXj + impulse * invM2 * nx;
    float newVelYj = oldVelYj + impulse * invM2 * ny;

    sys.velX[i] = newVelXi;
    sys.velY[i] = newVelYi;
    sys.velX[j] = newVelXj;
    sys.velY[j] = newVelYj;

    sys.posX[i] += remainingT * dt * (newVelXi - oldVelXi);
    sys.posY[i] += remainingT * dt * (newVelYi - oldVelYi);
    sys.posX[j] += remainingT * dt * (newVelXj - oldVelXj);
    sys.posY[j] += remainingT * dt * (newVelYj - oldVelYj);
}

inline void findAllCollisionCandidatesForRange(const ParticleSystem& sys, const SpatialTree& tree, CollisionCandidates& cc, int32_t startI, int32_t endI, float globalMaxRadius, float gap) {
    for (int32_t i = startI; i < endI; i++) {
        float speed = sqrtf(cc.origVelX[i] * cc.origVelX[i] + cc.origVelY[i] * cc.origVelY[i]);
        float searchRadius = sys.radius[i] + globalMaxRadius + gap + speed;
        int32_t* slot = cc.buffer + (size_t)i * MAX_CANDIDATES_PER_PARTICLE;
        cc.counts[i] = tree.findNearbyInto(sys, i, searchRadius, slot, MAX_CANDIDATES_PER_PARTICLE);
    }
}

// Full per-frame collision pass: build the collision tree, snapshot velocities, batch-search
// candidates across threads, then resolve swept collisions sequentially (this part can't be
// parallelized - resolveImpulse reads/writes live velocities and resolvedPairs has to stay
// consistent across the whole pass, exactly as in collide.ts). The web app's own profiling
// found the threaded search dominant; this native port's profiling at 50k particles instead
// found the single-threaded resolve loop dominant under a dense central-mass cluster, until
// PairSet's dedup-gating fix (see its header comment) brought it back in line - pass a
// CollideTimings pointer to get a build/search/resolve breakdown if re-checking this.
struct CollideTimings {
    double buildMs = 0, searchMs = 0, resolveMs = 0;
};

// `dt` is this call's substep length (1.0 for a full, un-substepped frame - see main.cpp's
// adaptive substepping). The swept-collision reconstruction below has to reconstruct actual
// *positions* (this substep's start-of-drift point, the contact point), which needs the
// real per-substep displacement (velocity*dt), not velocity itself - see relDriftX/Y.
inline void collideParticles(ParticleSystem& sys, SpatialTree& tree, CollisionCandidates& cc, float G, float softeningFactor, float restitution, float gap, int32_t maxThreads, float dt, CollideTimings* timings = nullptr) {
    int32_t count = sys.count;
    auto stageStart = std::chrono::steady_clock::now();

    tree.maxDepth = constants::COLLISION_TREE_MAX_DEPTH;
    tree.leafCapacity = constants::COLLISION_TREE_LEAF_CAPACITY;
    tree.build(sys, count);
    if (timings) {
        auto now = std::chrono::steady_clock::now();
        timings->buildMs = std::chrono::duration<double, std::milli>(now - stageStart).count();
        stageStart = now;
    }

    for (int32_t i = 0; i < count; i++) {
        cc.origVelX[i] = sys.velX[i];
        cc.origVelY[i] = sys.velY[i];
    }

    float globalMaxRadius = 0.f;
    for (int32_t i = 0; i < count; i++) {
        if (sys.radius[i] > globalMaxRadius) globalMaxRadius = sys.radius[i];
    }

    unsigned int hwThreads = std::thread::hardware_concurrency();
    int32_t threadCap = maxThreads < 1 ? 1 : maxThreads;
    int32_t threadCount = hwThreads < 1 ? 1 : (int32_t)std::min(hwThreads, (unsigned int)threadCap);
    if (threadCount > count) threadCount = count > 0 ? count : 1;

    if (threadCount <= 1) {
        findAllCollisionCandidatesForRange(sys, tree, cc, 0, count, globalMaxRadius, gap);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        int32_t chunk = (count + threadCount - 1) / threadCount;
        for (int32_t t = 0; t < threadCount; t++) {
            int32_t startI = t * chunk;
            int32_t endI = std::min(startI + chunk, count);
            if (startI >= endI) break;
            workers.emplace_back(findAllCollisionCandidatesForRange, std::ref(sys), std::ref(tree), std::ref(cc), startI, endI, globalMaxRadius, gap);
        }
        for (auto& w : workers) w.join();
    }
    if (timings) {
        auto now = std::chrono::steady_clock::now();
        timings->searchMs = std::chrono::duration<double, std::milli>(now - stageStart).count();
        stageStart = now;
    }

    // Packs an unordered pair (i,j) as one integer so a pair found from both directions
    // (i seeking j, j seeking i) only resolves once this frame - see collide.ts.
    cc.resolvedPairs.beginFrame();

    for (int32_t i = 0; i < count; i++) {
        int32_t candCount = cc.counts[i];
        int32_t* candStart = cc.buffer + (size_t)i * MAX_CANDIDATES_PER_PARTICLE;

        for (int32_t c = 0; c < candCount; c++) {
            int32_t j = candStart[c];

            float relVx = cc.origVelX[j] - cc.origVelX[i];
            float relVy = cc.origVelY[j] - cc.origVelY[i];
            // Actual displacement over this substep (not velocity itself, unless dt==1) -
            // the swept reconstruction below works entirely in position space.
            float relDriftX = relVx * dt;
            float relDriftY = relVy * dt;
            float relDriftSpeedSq = relDriftX * relDriftX + relDriftY * relDriftY;

            float endDx = sys.posX[j] - sys.posX[i];
            float endDy = sys.posY[j] - sys.posY[i];
            float startDx = endDx - relDriftX;
            float startDy = endDy - relDriftY;

            float touchDistance = sys.radius[i] + sys.radius[j] + gap;
            float cVal = startDx * startDx + startDy * startDy - touchDistance * touchDistance;

            float t, contactDx, contactDy;
            if (cVal <= 0.f) {
                t = 0.f;
                contactDx = startDx;
                contactDy = startDy;
            } else if (relDriftSpeedSq < 1e-9f) {
                continue;
            } else {
                float b = 2.f * (startDx * relDriftX + startDy * relDriftY);
                float discriminant = b * b - 4.f * relDriftSpeedSq * cVal;
                if (discriminant < 0.f) continue;
                t = (-b - sqrtf(discriminant)) / (2.f * relDriftSpeedSq);
                if (t < 0.f || t > 1.f) continue;
                contactDx = startDx + t * relDriftX;
                contactDy = startDy + t * relDriftY;
            }

            // Sign-only test (approaching vs separating) - dotting with true velocity or
            // with relDriftX/Y (velocity*dt, dt>0) always agrees on sign, so this can use
            // either; kept as velocity to match collide.ts's original formula directly.
            float closingSpeed = contactDx * relVx + contactDy * relVy;

            // Dedup only gates here, not the whole candidate loop - every branch above
            // (the two `continue`s and this closingSpeed sign test) is provably
            // direction-invariant (see this function's header comment for the algebra):
            // swapping which particle initiates the check negates
            // relVx/relVy/contactDx/contactDy together, which cancels out in every
            // quantity that actually drives a decision. So re-deriving a "no resolution"
            // outcome from the other particle's own candidate list costs a few flops and
            // no hash lookup - only an outcome that actually calls resolveOverlap/
            // resolveImpulse needs the once-per-frame guarantee a hash lookup provides.
            // Profiling at 50k particles showed this is where nearly all of the resolve
            // stage's cost was going: with dedup gating every examined candidate instead
            // of just real resolutions, per-candidate cost was ~2.5x higher.
            int64_t pairKey = i < j ? (int64_t)i * sys.capacity + j : (int64_t)j * sys.capacity + i;

            if (closingSpeed >= 0.f) {
                if (!cc.resolvedPairs.insertIfNew(pairKey)) continue;
                resolveOverlap(sys, i, j, G, softeningFactor, gap);
                continue;
            }

            if (!cc.resolvedPairs.insertIfNew(pairKey)) continue;

            float dist = sqrtf(contactDx * contactDx + contactDy * contactDy);
            if (dist == 0.f) dist = 1e-6f;
            float nx = contactDx / dist;
            float ny = contactDy / dist;

            resolveImpulse(sys, i, j, nx, ny, 1.f - t, restitution, dt);
            resolveOverlap(sys, i, j, G, softeningFactor, gap);
        }
    }
    if (timings) {
        timings->resolveMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
    }
}
