@echo off
setlocal

rem Compiles the native C++/OpenGL port (see src\main.cpp) - no CMake in this MSYS2 UCRT64
rem toolchain, so a flat g++ invocation is simplest. Run from anywhere (double-click included);
rem paths below are relative to this script's own location (%~dp0).
rem
rem g++ isn't on the normal Windows PATH (it only lives under MSYS2's install), so this adds
rem it just for this script's process - edit MINGW_BIN below if your MSYS2 lives somewhere
rem other than the default C:\msys64.
set MINGW_BIN=C:\msys64\ucrt64\bin
set PATH=%MINGW_BIN%;%PATH%

where g++ >nul 2>nul
if errorlevel 1 (
    echo g++ not found - edit MINGW_BIN in this script to point at your MSYS2 ucrt64\bin.
    exit /b 1
)

rem -static (+ -DGLEW_STATIC, which has to match: GLEW's headers assume DLL-imported symbols
rem unless told otherwise) links glfw3/glew32/libgcc/libstdc++/libwinpthread directly into the
rem exe, so the result depends on nothing but core Windows DLLs and can run outside any MSYS2
rem shell - without it, the exe silently needs glfw3.dll/glew32.dll/libstdc++-6.dll/etc from
rem ucrt64\bin on PATH at runtime, which won't be true for a plain double-click.
g++ -std=c++17 -O3 -march=native -static -pthread -DGLEW_STATIC ^
    "%~dp0src\main.cpp" ^
    -o "%~dp0nebula_native.exe" ^
    -lglfw3 -lglew32 -lopengl32 -lgdi32

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo Built %~dp0nebula_native.exe
