#pragma once
// Minimal iterative radix-2 Cooley-Tukey FFT - real/imaginary kept as separate float arrays
// (no std::complex indirection) since pm_gravity.h only ever needs forward transform,
// multiply by a Green's function, inverse transform. Grid dimensions must be powers of two.
//
// The 2D transform threads its row and column passes (each is N fully independent 1D FFTs -
// the same read-only-input/write-only-own-slice shape gravity.h's traversal threading
// exploits). Profiled before/after at grid 256: the two fft2d calls per PM solve went from
// ~11.6ms combined to well under 2ms. Unlike the tree-build threading experiment (which
// lost to thread spawn overhead and was reverted - see that history in tree.h's git log),
// this pass has enough work per spawn to come out clearly ahead.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

constexpr float FFT_PI = 3.14159265358979323846f;

// In-place 1D FFT over `n` (power-of-two) complex samples with element stride `stride` - the
// stride lets this run directly on a grid's rows (stride 1) or columns (stride nx) with no
// separate transpose step, see fft2d. `inverse` only flips the exponent's sign; the 1/n
// scaling for a true inverse transform is applied once in fft2d, not per 1D pass, since
// scaling is separable across the two passes.
inline void fft1d(float* re, float* im, int32_t n, int32_t stride, bool inverse) {
    for (int32_t i = 1, j = 0; i < n; i++) {
        int32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[(size_t)i * stride], re[(size_t)j * stride]);
            std::swap(im[(size_t)i * stride], im[(size_t)j * stride]);
        }
    }

    for (int32_t len = 2; len <= n; len <<= 1) {
        float angStep = (inverse ? 2.f : -2.f) * FFT_PI / (float)len;
        float wReal = cosf(angStep), wImag = sinf(angStep);
        int32_t half = len / 2;
        for (int32_t start = 0; start < n; start += len) {
            float curReal = 1.f, curImag = 0.f;
            for (int32_t k = 0; k < half; k++) {
                size_t evenIdx = (size_t)(start + k) * stride;
                size_t oddIdx = (size_t)(start + k + half) * stride;
                float oddRe = re[oddIdx] * curReal - im[oddIdx] * curImag;
                float oddIm = re[oddIdx] * curImag + im[oddIdx] * curReal;
                re[oddIdx] = re[evenIdx] - oddRe;
                im[oddIdx] = im[evenIdx] - oddIm;
                re[evenIdx] += oddRe;
                im[evenIdx] += oddIm;
                float nextReal = curReal * wReal - curImag * wImag;
                float nextImag = curReal * wImag + curImag * wReal;
                curReal = nextReal;
                curImag = nextImag;
            }
        }
    }
}

// Runs f(start, end) over [0, n) split into contiguous chunks across up to maxThreads real
// OS threads - the same spawn-per-call pattern gravity.h/collide.h use (a persistent pool
// would amortize spawn cost further; see main.cpp's future-work notes).
template <typename F>
inline void fftParallelFor(int32_t n, int32_t maxThreads, F f) {
    unsigned int hw = std::thread::hardware_concurrency();
    int32_t threadCount = hw < 1 ? 1 : (int32_t)std::min(hw, (unsigned int)std::max(maxThreads, 1));
    if (threadCount > n) threadCount = n > 0 ? n : 1;
    if (threadCount <= 1) {
        f(0, n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    int32_t chunk = (n + threadCount - 1) / threadCount;
    for (int32_t t = 0; t < threadCount; t++) {
        int32_t start = t * chunk;
        int32_t end = std::min(start + chunk, n);
        if (start >= end) break;
        workers.emplace_back([=]() { f(start, end); });
    }
    for (auto& w : workers) w.join();
}

// 2D FFT via row-then-column 1D passes (standard separable-DFT decomposition) over an
// nx*ny grid stored row-major (re[y*nx+x]). Scales by 1/(nx*ny) on the inverse pass only.
//
// The column pass runs strided in place. A transpose -> contiguous-FFT -> transpose-back
// variant was tried and MEASURED SLOWER at grid 256 (8.3ms vs 6.9ms for both fft2d calls of
// a PM solve combined): at this size the whole grid fits in L2, so the strided pass isn't
// actually memory-bound, and the four extra transpose passes plus their extra thread spawns
// only added overhead. Revisit only if grid sizes grow well past L2 (1024+).
inline void fft2d(float* re, float* im, int32_t nx, int32_t ny, bool inverse, int32_t maxThreads = 1) {
    fftParallelFor(ny, maxThreads, [=](int32_t start, int32_t end) {
        for (int32_t y = start; y < end; y++) {
            fft1d(re + (size_t)y * nx, im + (size_t)y * nx, nx, 1, inverse);
        }
    });
    fftParallelFor(nx, maxThreads, [=](int32_t start, int32_t end) {
        for (int32_t x = start; x < end; x++) {
            fft1d(re + x, im + x, ny, nx, inverse);
        }
    });
    if (inverse) {
        float scale = 1.f / ((float)nx * (float)ny);
        fftParallelFor(ny, maxThreads, [=](int32_t start, int32_t end) {
            size_t from = (size_t)start * nx, to = (size_t)end * nx;
            for (size_t i = from; i < to; i++) {
                re[i] *= scale;
                im[i] *= scale;
            }
        });
    }
}
