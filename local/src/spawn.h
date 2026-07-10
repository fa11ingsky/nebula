#pragma once
// Native port of src/lib/sim/spawn.ts, scoped to this benchmark's needs: the swarm plus an
// optional single fixed central mass. Every particle (and the central mass) renders plain
// white - mergingEnabled-based/mass-gradient coloring (colors.h) isn't used here.
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "constants.h"
#include "particle_system.h"

struct RandomSpawnPoint {
    float x, y;
};

inline RandomSpawnPoint randomSpawnPoint(std::mt19937& rng, float centerX, float centerY, float radius) {
    constexpr float TWO_PI = 6.28318530718f;
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    float angle = unit(rng) * TWO_PI;
    float r = radius * sqrtf(unit(rng));
    return {centerX + r * cosf(angle), centerY + r * sinf(angle)};
}

// Rigid-body-style rotation about the system's own center of mass - see spawn.ts's
// initializeAngularMomentum for the full derivation of why this gives exactly zero net
// linear momentum and exactly targetL net angular momentum. A fixed particle (the optional
// central mass) is left at velocity 0, same reasoning as the web app.
inline void initializeAngularMomentum(ParticleSystem& sys, float targetL) {
    float comX, comY;
    computeCenterOfMass(sys, comX, comY);

    double momentOfInertia = 0.0;
    std::vector<float> offsetX(sys.count), offsetY(sys.count);
    for (int32_t i = 0; i < sys.count; i++) {
        if (sys.fixed[i]) continue;
        float dx = sys.posX[i] - comX;
        float dy = sys.posY[i] - comY;
        offsetX[i] = dx;
        offsetY[i] = dy;
        momentOfInertia += (double)sys.mass[i] * (dx * dx + dy * dy);
    }

    if (momentOfInertia < 1e-6) return;

    float omega = (float)(targetL / momentOfInertia);
    for (int32_t i = 0; i < sys.count; i++) {
        if (sys.fixed[i]) continue;
        sys.velX[i] = -omega * offsetY[i];
        sys.velY[i] = omega * offsetX[i];
    }
}

// Builds the swarm (totalParticles bodies in a disc) plus, if requested, one additional
// fixed particle at the exact center holding CENTRAL_MASS_FRACTION of MAX_MASS - sized like
// an ordinary swarm particle (not sqrt(centralMass)) so it reads as one more body rather
// than a giant sphere, matching spawn.ts's radiusOverride.
inline void spawnParticles(ParticleSystem& sys, int32_t totalParticles, bool includeCentralMass, float worldWidth, float worldHeight, std::mt19937& rng) {
    int32_t capacity = totalParticles + (includeCentralMass ? 1 : 0);
    sys.allocate(capacity);

    float centerX = worldWidth / 2.f;
    float centerY = worldHeight / 2.f;
    float spawnRadius = std::min(worldWidth, worldHeight) / 2.f * constants::SPAWN_RADIUS_FRACTION;
    float particleMass = constants::MAX_MASS / (float)totalParticles;
    float particleRadius = sqrtf(particleMass);

    for (int32_t i = 0; i < totalParticles; i++) {
        RandomSpawnPoint p = randomSpawnPoint(rng, centerX, centerY, spawnRadius);
        addParticle(sys, p.x, p.y, particleMass, 1.f, 1.f, 1.f);
    }

    if (includeCentralMass) {
        float centralMass = constants::MAX_MASS * constants::CENTRAL_MASS_FRACTION;
        addParticle(sys, centerX, centerY, centralMass, 1.f, 1.f, 1.f, true, particleRadius);
    }

    initializeAngularMomentum(sys, constants::TOTAL_ANGULAR_MOMENTUM);
}
