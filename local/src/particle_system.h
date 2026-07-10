#pragma once
// Mirrors src/lib/sim/particleSystem.ts's struct-of-arrays particle store - a native,
// fixed-capacity version, since this benchmark never merges/removes particles (matching
// the web app's "2d, non-merging" collision-mode focus this whole native port exists to
// stress-test). No dynamic growth needed as a result.
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct ParticleSystem {
    int32_t capacity = 0;
    int32_t count = 0;
    float* posX = nullptr;
    float* posY = nullptr;
    float* velX = nullptr;
    float* velY = nullptr;
    float* accX = nullptr;
    float* accY = nullptr;
    float* mass = nullptr;
    float* radius = nullptr;
    float* colorR = nullptr; // 0..1 (OpenGL convention, unlike the web app's 0..255)
    float* colorG = nullptr;
    float* colorB = nullptr;
    // 1 for a particle that never moves (see spawn.ts's fixed central mass) - kickAll/
    // driftAll skip it, and collide.cpp's resolution treats it as infinitely massive, the
    // same design as particleSystem.ts's `fixed` field.
    uint8_t* fixed = nullptr;

    void allocate(int32_t cap) {
        capacity = cap;
        count = 0;
        posX = new float[cap];
        posY = new float[cap];
        velX = new float[cap];
        velY = new float[cap];
        accX = new float[cap];
        accY = new float[cap];
        mass = new float[cap];
        radius = new float[cap];
        colorR = new float[cap];
        colorG = new float[cap];
        colorB = new float[cap];
        fixed = new uint8_t[cap];
    }
};

inline int32_t addParticle(ParticleSystem& sys, float x, float y, float mass, float r, float g, float b, bool isFixed = false, float radiusOverride = -1.f) {
    int32_t i = sys.count;
    sys.posX[i] = x;
    sys.posY[i] = y;
    sys.velX[i] = 0.f;
    sys.velY[i] = 0.f;
    sys.accX[i] = 0.f;
    sys.accY[i] = 0.f;
    sys.mass[i] = mass;
    sys.radius[i] = radiusOverride >= 0.f ? radiusOverride : sqrtf(mass);
    sys.colorR[i] = r;
    sys.colorG[i] = g;
    sys.colorB[i] = b;
    sys.fixed[i] = isFixed ? 1 : 0;
    sys.count++;
    return i;
}

inline void kickAll(ParticleSystem& sys, float dt) {
    for (int32_t i = 0; i < sys.count; i++) {
        if (sys.fixed[i]) continue;
        sys.velX[i] += sys.accX[i] * dt;
        sys.velY[i] += sys.accY[i] * dt;
    }
}

inline void driftAll(ParticleSystem& sys, float dt) {
    for (int32_t i = 0; i < sys.count; i++) {
        if (sys.fixed[i]) continue;
        sys.posX[i] += sys.velX[i] * dt;
        sys.posY[i] += sys.velY[i] * dt;
    }
}

inline void resetAccelerationAll(ParticleSystem& sys) {
    for (int32_t i = 0; i < sys.count; i++) {
        sys.accX[i] = 0.f;
        sys.accY[i] = 0.f;
    }
}

inline void computeCenterOfMass(const ParticleSystem& sys, float& outX, float& outY) {
    double totalMass = 0.0, x = 0.0, y = 0.0;
    for (int32_t i = 0; i < sys.count; i++) {
        totalMass += sys.mass[i];
        x += (double)sys.mass[i] * sys.posX[i];
        y += (double)sys.mass[i] * sys.posY[i];
    }
    outX = totalMass > 0 ? (float)(x / totalMass) : 0.f;
    outY = totalMass > 0 ? (float)(y / totalMass) : 0.f;
}
