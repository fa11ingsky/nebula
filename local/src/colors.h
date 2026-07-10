#pragma once
// Port of src/lib/sim/colors.ts's mass -> color gradient (brown -> blue -> green -> red ->
// yellow -> white as mass climbs toward maxMass), outputting 0..1 floats (OpenGL/GLSL
// convention) instead of the web app's 0..255 bytes.
#include <algorithm>
#include <cmath>

struct ColorStop {
    float r, g, b;
};

inline ColorStop interpolateColorStops(const ColorStop* stops, int stopCount, float t) {
    float clampedT = std::min(std::max(t, 0.f), 1.f);
    int segmentCount = stopCount - 1;
    float scaled = clampedT * segmentCount;
    int index = std::min((int)scaled, segmentCount - 1);
    float localT = scaled - index;

    const ColorStop& a = stops[index];
    const ColorStop& b = stops[index + 1];
    return {
        a.r + (b.r - a.r) * localT,
        a.g + (b.g - a.g) * localT,
        a.b + (b.b - a.b) * localT,
    };
}

inline ColorStop getColorForMass(float mass, float maxMass) {
    static const ColorStop stops[] = {
        {139.f / 255.f, 69.f / 255.f, 19.f / 255.f},   // brown
        {59.f / 255.f, 130.f / 255.f, 246.f / 255.f},  // blue
        {50.f / 255.f, 200.f / 255.f, 90.f / 255.f},   // green
        {220.f / 255.f, 40.f / 255.f, 40.f / 255.f},   // red
        {255.f / 255.f, 221.f / 255.f, 51.f / 255.f},  // yellow
        {1.f, 1.f, 1.f},                                // white
    };
    return interpolateColorStops(stops, 6, mass / maxMass);
}
