// Minimal iterative radix-2 Cooley-Tukey FFT - a direct port of the native app's
// local/src/fft.h, with real/imaginary kept as separate Float32Arrays (no complex-number
// objects) since pmGravity.ts only ever needs forward transform, multiply by a Green's
// function, inverse transform. Grid dimensions must be powers of two.
//
// Single-threaded, unlike the native version's row/column thread split: this app
// deliberately runs without SharedArrayBuffer (see simulation.worker.ts's header comment
// on cross-origin isolation and GitHub Pages), so there's no way to hand slices of these
// arrays to other workers without per-frame copies that would eat the parallelism win.
// The CPU PM path this backs is the moderate-particle-count option; the WebGPU pipeline
// (webgpuSim.ts) is the one that scales.

/**
 * In-place 1D FFT over `n` (power-of-two) complex samples with element stride `stride` -
 * the stride lets this run directly on a grid's rows (stride 1) or columns (stride nx)
 * with no separate transpose step, see fft2d. `inverse` only flips the exponent's sign;
 * the 1/n scaling for a true inverse transform is applied once in fft2d, not per 1D pass,
 * since scaling is separable across the two passes.
 */
export function fft1d(re, im, offset, n, stride, inverse) {
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            const ii = offset + i * stride;
            const jj = offset + j * stride;
            const tr = re[ii]; re[ii] = re[jj]; re[jj] = tr;
            const ti = im[ii]; im[ii] = im[jj]; im[jj] = ti;
        }
    }

    for (let len = 2; len <= n; len <<= 1) {
        const angStep = (inverse ? 2 : -2) * Math.PI / len;
        const wReal = Math.cos(angStep), wImag = Math.sin(angStep);
        const half = len / 2;
        for (let start = 0; start < n; start += len) {
            let curReal = 1, curImag = 0;
            for (let k = 0; k < half; k++) {
                const evenIdx = offset + (start + k) * stride;
                const oddIdx = offset + (start + k + half) * stride;
                const oddRe = re[oddIdx] * curReal - im[oddIdx] * curImag;
                const oddIm = re[oddIdx] * curImag + im[oddIdx] * curReal;
                re[oddIdx] = re[evenIdx] - oddRe;
                im[oddIdx] = im[evenIdx] - oddIm;
                re[evenIdx] += oddRe;
                im[evenIdx] += oddIm;
                const nextReal = curReal * wReal - curImag * wImag;
                const nextImag = curReal * wImag + curImag * wReal;
                curReal = nextReal;
                curImag = nextImag;
            }
        }
    }
}

/**
 * 2D FFT via row-then-column 1D passes (standard separable-DFT decomposition) over an
 * nx*ny grid stored row-major (re[y*nx+x]). Scales by 1/(nx*ny) on the inverse pass only.
 * The column pass runs strided in place - the native version measured a transpose-based
 * variant slower at these grid sizes (fits in L2), so this keeps the same layout.
 */
export function fft2d(re, im, nx, ny, inverse) {
    for (let y = 0; y < ny; y++) {
        fft1d(re, im, y * nx, nx, 1, inverse);
    }
    for (let x = 0; x < nx; x++) {
        fft1d(re, im, x, ny, nx, inverse);
    }
    if (inverse) {
        const scale = 1 / (nx * ny);
        const total = nx * ny;
        for (let i = 0; i < total; i++) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}
