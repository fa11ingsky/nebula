#!/usr/bin/env bash
# Compiles the native C++/OpenGL port (see src/main.cpp) - no CMake available in this
# MSYS2 UCRT64 toolchain, so a flat g++ invocation is simplest. Run from anywhere; paths
# below are relative to this script's own directory.
#
# -static (+ -DGLEW_STATIC, which has to match: GLEW's headers assume DLL-imported symbols
# unless told otherwise) links glfw3/glew32/libgcc/libstdc++/libwinpthread all directly into
# the exe - confirmed via ldd that the result depends on nothing but core Windows DLLs
# (opengl32/gdi32/user32/etc, present on every install). Without this, the exe silently
# requires glfw3.dll/glew32.dll/libstdc++-6.dll/etc from MSYS2's ucrt64/bin on PATH at
# runtime - fine from this same shell, but breaks the moment it's run from anywhere else
# (build.bat, double-click, a machine without MSYS2 installed at all).
set -e
cd "$(dirname "$0")"

g++ -std=c++17 -O3 -march=native -static -pthread -DGLEW_STATIC \
    src/main.cpp \
    -o nebula_native.exe \
    -lglfw3 -lglew32 -lopengl32 -lgdi32

echo "Built local/nebula_native.exe"
