#pragma once
// Screen-space overlay renderer for the debug panel: translucent background rect + bitmap-
// font text (font.h), drawn as instanced axis-aligned quads on top of the particle scene.
// Same pixel-coordinate convention as renderer.h (origin top-left, y grows downward) so
// panel placement lines up with the particle world without any extra transform.
#include <GL/glew.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "font.h"

struct UiQuadInstance {
    float x, y, w, h;
    float r, g, b, a;
};

struct UiRenderer {
    GLuint shaderProgram = 0;
    GLuint vao = 0;
    GLuint quadVbo = 0;
    GLuint instanceVbo = 0;
    GLint viewportSizeLoc = -1;
    std::vector<UiQuadInstance> instances;
    int32_t reservedCapacity = 0;

    static GLuint compileShader(GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            fprintf(stderr, "UI shader compile error: %s\n", log);
            exit(1);
        }
        return shader;
    }

    void init() {
        const char* vertSrc = R"GLSL(
            #version 330 core
            layout(location = 0) in vec2 localPos;
            layout(location = 1) in vec4 instanceRect; // x, y, w, h in screen pixels
            layout(location = 2) in vec4 instanceColor;
            uniform vec2 viewportSize;
            out vec4 fragColor;
            void main() {
                vec2 worldPos = instanceRect.xy + localPos * instanceRect.zw;
                vec2 ndc = (worldPos / viewportSize) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                fragColor = instanceColor;
            }
        )GLSL";
        const char* fragSrc = R"GLSL(
            #version 330 core
            in vec4 fragColor;
            out vec4 outColor;
            void main() {
                outColor = fragColor;
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
            fprintf(stderr, "UI program link error: %s\n", log);
            exit(1);
        }
        glDeleteShader(vert);
        glDeleteShader(frag);
        viewportSizeLoc = glGetUniformLocation(shaderProgram, "viewportSize");

        float quadVerts[12] = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &quadVbo);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &instanceVbo);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(UiQuadInstance), (void*)0);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(UiQuadInstance), (void*)(4 * sizeof(float)));
        glVertexAttribDivisor(2, 1);

        glBindVertexArray(0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    void begin() {
        instances.clear();
    }

    void addQuad(float x, float y, float w, float h, float r, float g, float b, float a) {
        instances.push_back({x, y, w, h, r, g, b, a});
    }

    // Draws text at (x, y) (top-left corner of the first glyph), each font pixel scaled to
    // `pixelSize` screen pixels, 1-pixel gaps between font pixels and 1-pixel-scaled gaps
    // between characters - see font.h for the 5x7 glyph data.
    void addText(float x, float y, float pixelSize, const char* text, float r, float g, float b, float a) {
        float cursorX = x;
        for (const char* p = text; *p; p++) {
            const Glyph* glyph = getGlyph(*p);
            for (int row = 0; row < 7; row++) {
                uint8_t bits = glyph->rows[row];
                for (int col = 0; col < 5; col++) {
                    if (bits & (1 << (4 - col))) {
                        addQuad(cursorX + col * pixelSize, y + row * pixelSize, pixelSize, pixelSize, r, g, b, a);
                    }
                }
            }
            cursorX += 6.f * pixelSize; // 5 columns + 1 column of spacing
        }
    }

    static float textWidth(const char* text, float pixelSize) {
        return (float)strlen(text) * 6.f * pixelSize;
    }

    void flush(int32_t viewportWidth, int32_t viewportHeight) {
        if (instances.empty()) return;
        if ((int32_t)instances.size() > reservedCapacity) {
            reservedCapacity = (int32_t)instances.size() * 2;
            glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
            glBufferData(GL_ARRAY_BUFFER, reservedCapacity * sizeof(UiQuadInstance), nullptr, GL_DYNAMIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, instances.size() * sizeof(UiQuadInstance), instances.data());

        glUseProgram(shaderProgram);
        glUniform2f(viewportSizeLoc, (float)viewportWidth, (float)viewportHeight);
        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)instances.size());
        glBindVertexArray(0);
    }
};
