#pragma once
// Flat-color instanced circle renderer - no textures/explosion-flash effects (this port is
// scoped to the physics benchmark, not a visual match of Particles.vue). Each particle is a
// small triangle-fan disc, drawn once per particle via glDrawArraysInstanced: per-particle
// radius+color are uploaded once (they never change post-spawn, since merging isn't ported),
// per-particle position (and density - see below) is re-uploaded every frame.
//
// Particles in dense regions render as larger, softer-edged, lower-peak-alpha blobs instead
// of small hard-edged discs - overlapping soft blobs blend together into a hazy, blurred-
// looking mass exactly where the swarm is crowded, while an isolated particle still renders
// as a crisp small disc (density ~0 reduces to the original hard-edged look). `density` is
// caller-supplied per particle, already normalized to 0..1 - main.cpp reuses collide.h's own
// per-particle candidate count (already computed every frame for collision) as a free local-
// density proxy, so this costs no extra spatial query.
//
// Position (vec2) and density (float) are SEPARATE per-instance buffers rather than one
// interleaved vec3 - deliberately matching the buffer layout gpu_sim.h's compute pipeline
// produces, so GPU mode renders straight from the physics SSBOs (renderFromBuffers) with
// zero per-frame copies or CPU round trips. The CPU path just uploads two arrays instead
// of one interleaved one.
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "particle_system.h"

struct Renderer {
    GLuint shaderProgram = 0;
    GLuint vao = 0;
    GLuint circleVbo = 0;      // unit-circle triangle-fan mesh, shared by every instance
    GLuint posVbo = 0;         // per-instance position (vec2), re-uploaded every frame (CPU path)
    GLuint densityVbo = 0;     // per-instance density (float), re-uploaded every frame (CPU path)
    GLuint staticVbo = 0;      // per-instance radius+color, uploaded once
    GLuint externalVao = 0;    // lazily-built VAO over caller-owned buffers - see renderFromBuffers
    GLuint externalPosBuf = 0, externalDensityBuf = 0; // which buffers externalVao was built for
    int32_t circleVertexCount = 0;
    GLint viewportSizeLoc = -1;
    GLint zoomLoc = -1;
    GLint cameraOffsetLoc = -1;
    GLint densityColorModeLoc = -1;
    bool densityColors = false; // --density-colors: color particles by local density instead of their own color

    static GLuint compileShader(GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            fprintf(stderr, "Shader compile error: %s\n", log);
            exit(1);
        }
        return shader;
    }

    void init(int32_t particleCapacity) {
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            fprintf(stderr, "glewInit failed\n");
            exit(1);
        }

        const char* vertSrc = R"GLSL(
            #version 330 core
            layout(location = 0) in vec2 localPos;
            layout(location = 1) in vec2 instancePos;
            layout(location = 2) in float instanceRadius;
            layout(location = 3) in vec3 instanceColor;
            layout(location = 4) in float instanceDensity; // 0..1
            uniform vec2 viewportSize;
            uniform float zoom;
            uniform vec2 cameraOffset; // world point that maps to the center of the screen
            uniform int densityColorMode; // --density-colors: color by local density instead of instanceColor
            out vec3 fragColor;
            out vec2 fragLocalPos;
            out float fragDensity;

            // Heat-style ramp over the same 0..1 density that drives the blur: isolated
            // particles read cool blue, mid-density regions red, and packed cores climb
            // through yellow into white - so temperature-of-color tracks crowding.
            vec3 densityRamp(float t) {
                vec3 blue = vec3(0.25, 0.45, 1.0);
                vec3 red = vec3(0.9, 0.15, 0.1);
                vec3 yellow = vec3(1.0, 0.85, 0.2);
                vec3 white = vec3(1.0, 1.0, 1.0);
                if (t < 0.3) return mix(blue, red, t / 0.5);
                if (t < 0.6) return mix(red, yellow, (t - 0.5) / 0.3);
                return mix(yellow, white, (t - 0.8) / 0.2);
            }

            void main() {
                float density = instanceDensity;
                // Denser regions render as a bigger soft blob, up to 2.5x the base radius -
                // this is what actually lets neighboring blobs' soft edges overlap and merge
                // into a blur, rather than just each particle fading out in isolation.
                float sizeScale = 1.0 + density * 1.5;
                vec2 worldPos = instancePos + localPos * instanceRadius * sizeScale;
                // Camera transform: cameraOffset is the world point currently centered on
                // screen (click-and-drag pans it - see main.cpp's cursor-pos callback), zoom
                // scales distances from that point (scroll-to-cursor keeps the world point
                // under the cursor fixed across a zoom change - see main.cpp's scroll
                // callback, which does the equivalent math on the CPU side to update
                // cameraOffset before this uniform is ever uploaded).
                vec2 screenCenter = viewportSize * 0.5;
                vec2 zoomed = (worldPos - cameraOffset) * zoom + screenCenter;
                vec2 ndc = (zoomed / viewportSize) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                fragColor = densityColorMode != 0 ? densityRamp(density) : instanceColor;
                fragLocalPos = localPos;
                fragDensity = density;
            }
        )GLSL";

        const char* fragSrc = R"GLSL(
            #version 330 core
            in vec3 fragColor;
            in vec2 fragLocalPos;
            in float fragDensity;
            out vec4 outColor;
            void main() {
                float dist = length(fragLocalPos);
                // At density 0: a near-hard edge at the nominal radius (matches the old
                // flat-disc look exactly). At density 1: the edge softens all the way back
                // to the center, so the whole (already-enlarged, see vertex shader) blob is
                // one smooth radial falloff instead of a disc - a blur, not just a bigger dot.
                float edgeStart = mix(0.92, 0.0, fragDensity);
                float alpha = 1.0 - smoothstep(edgeStart, 1.0, dist);
                // Peak alpha also drops as density rises, so overlapping soft blobs blend
                // and accumulate via alpha compositing into a hazy mass instead of just
                // stacking as flat, fully-opaque coverage.
                alpha *= mix(1.0, 0.35, fragDensity);
                outColor = vec4(fragColor, alpha);
            }
        )GLSL";

        GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
        GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vert);
        glAttachShader(shaderProgram, frag);
        glLinkProgram(shaderProgram);
        GLint ok = 0;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(shaderProgram, sizeof(log), nullptr, log);
            fprintf(stderr, "Program link error: %s\n", log);
            exit(1);
        }
        glDeleteShader(vert);
        glDeleteShader(frag);
        viewportSizeLoc = glGetUniformLocation(shaderProgram, "viewportSize");
        zoomLoc = glGetUniformLocation(shaderProgram, "zoom");
        cameraOffsetLoc = glGetUniformLocation(shaderProgram, "cameraOffset");
        densityColorModeLoc = glGetUniformLocation(shaderProgram, "densityColorMode");

        // Unit-radius triangle fan: center, then N perimeter points (last repeats the first).
        const int32_t segments = 16;
        constexpr float TWO_PI = 6.28318530718f;
        std::vector<float> circleVerts;
        circleVerts.push_back(0.f);
        circleVerts.push_back(0.f);
        for (int32_t s = 0; s <= segments; s++) {
            float angle = (float)s / (float)segments * TWO_PI;
            circleVerts.push_back(cosf(angle));
            circleVerts.push_back(sinf(angle));
        }
        circleVertexCount = (int32_t)(circleVerts.size() / 2);

        glGenBuffers(1, &circleVbo);
        glBindBuffer(GL_ARRAY_BUFFER, circleVbo);
        glBufferData(GL_ARRAY_BUFFER, circleVerts.size() * sizeof(float), circleVerts.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &posVbo);
        glBindBuffer(GL_ARRAY_BUFFER, posVbo);
        glBufferData(GL_ARRAY_BUFFER, (size_t)particleCapacity * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &densityVbo);
        glBindBuffer(GL_ARRAY_BUFFER, densityVbo);
        glBufferData(GL_ARRAY_BUFFER, (size_t)particleCapacity * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        // radius (1 float) + color (3 floats) packed together, per instance.
        glGenBuffers(1, &staticVbo);
        glBindBuffer(GL_ARRAY_BUFFER, staticVbo);
        glBufferData(GL_ARRAY_BUFFER, (size_t)particleCapacity * 4 * sizeof(float), nullptr, GL_STATIC_DRAW);

        glGenVertexArrays(1, &vao);
        setupInstanceVao(vao, posVbo, densityVbo);

        glClearColor(0.02f, 0.02f, 0.04f, 1.f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // Configures a VAO over the given position (vec2/instance) and density (float/instance)
    // buffers, plus this renderer's own circle mesh and static radius/color buffer. Used
    // both for the CPU path's own VBOs and for GPU mode's physics SSBOs (any GL buffer
    // object can serve as a vertex source - renderFromBuffers relies on exactly that).
    void setupInstanceVao(GLuint targetVao, GLuint posBuffer, GLuint densityBuffer) {
        glBindVertexArray(targetVao);

        glBindBuffer(GL_ARRAY_BUFFER, circleVbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glVertexAttribDivisor(1, 1);

        glBindBuffer(GL_ARRAY_BUFFER, staticVbo);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glVertexAttribDivisor(2, 1);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(1 * sizeof(float)));
        glVertexAttribDivisor(3, 1);

        glBindBuffer(GL_ARRAY_BUFFER, densityBuffer);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glVertexAttribDivisor(4, 1);

        glBindVertexArray(0);
    }

    // Uploaded once after spawn - radius/color never change post-spawn in this port (no
    // merging, no growth), so there's no reason to re-upload them every frame.
    void uploadStaticInstanceData(const ParticleSystem& sys) {
        std::vector<float> data((size_t)sys.count * 4);
        for (int32_t i = 0; i < sys.count; i++) {
            data[i * 4 + 0] = sys.radius[i];
            data[i * 4 + 1] = sys.colorR[i];
            data[i * 4 + 2] = sys.colorG[i];
            data[i * 4 + 3] = sys.colorB[i];
        }
        glBindBuffer(GL_ARRAY_BUFFER, staticVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(float), data.data());
    }

    void drawInstances(GLuint sourceVao, int32_t count, int32_t framebufferWidth, int32_t framebufferHeight,
                        float worldWidth, float worldHeight, float zoom, float cameraOffsetX, float cameraOffsetY) {
        glViewport(0, 0, framebufferWidth, framebufferHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glUniform2f(viewportSizeLoc, worldWidth, worldHeight);
        glUniform1f(zoomLoc, zoom);
        glUniform2f(cameraOffsetLoc, cameraOffsetX, cameraOffsetY);
        glUniform1i(densityColorModeLoc, densityColors ? 1 : 0);
        glBindVertexArray(sourceVao);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, circleVertexCount, count);
        glBindVertexArray(0);
    }

    // `density` is one normalized (0..1) value per particle, indexed the same as sys - pass
    // nullptr to render everyone at density 0 (the original hard-edged look). `cameraOffsetX/Y`
    // is the world point mapped to the center of the screen (pans via click-and-drag) and
    // `zoom` scales distances from it (scroll wheel, pivoting on the cursor) - see
    // main.cpp's CameraState and its GLFW callbacks.
    void render(const ParticleSystem& sys, int32_t framebufferWidth, int32_t framebufferHeight, float worldWidth, float worldHeight, const float* density = nullptr, float zoom = 1.f, float cameraOffsetX = 0.f, float cameraOffsetY = 0.f) {
        std::vector<float> posData((size_t)sys.count * 2);
        for (int32_t i = 0; i < sys.count; i++) {
            posData[i * 2 + 0] = sys.posX[i];
            posData[i * 2 + 1] = sys.posY[i];
        }
        glBindBuffer(GL_ARRAY_BUFFER, posVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, posData.size() * sizeof(float), posData.data());

        std::vector<float> densityData((size_t)sys.count, 0.f);
        if (density) {
            std::copy(density, density + sys.count, densityData.begin());
        }
        glBindBuffer(GL_ARRAY_BUFFER, densityVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, densityData.size() * sizeof(float), densityData.data());

        drawInstances(vao, sys.count, framebufferWidth, framebufferHeight, worldWidth, worldHeight, zoom, cameraOffsetX, cameraOffsetY);
    }

    // GPU-physics path: renders straight from caller-owned buffers (gpu_sim.h's position and
    // density SSBOs) - no per-frame CPU copies at all. The external VAO is (re)built lazily
    // only if the buffer handles change.
    void renderFromBuffers(GLuint posBuffer, GLuint densityBuffer, int32_t count, int32_t framebufferWidth, int32_t framebufferHeight,
                            float worldWidth, float worldHeight, float zoom, float cameraOffsetX, float cameraOffsetY) {
        if (externalVao == 0 || externalPosBuf != posBuffer || externalDensityBuf != densityBuffer) {
            if (externalVao == 0) glGenVertexArrays(1, &externalVao);
            setupInstanceVao(externalVao, posBuffer, densityBuffer);
            externalPosBuf = posBuffer;
            externalDensityBuf = densityBuffer;
        }
        drawInstances(externalVao, count, framebufferWidth, framebufferHeight, worldWidth, worldHeight, zoom, cameraOffsetX, cameraOffsetY);
    }
};
