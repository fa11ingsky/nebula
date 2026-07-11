// Mass -> color logic, used by both particle rendering and explosion flashes.
import constants from '../constants.ts';

/**
 * Linearly interpolates through a sequence of [r,g,b] stops based on t (0..1), picking
 * whichever segment t falls into. Shared by the particle mass gradient and the explosion
 * mass gradient so both read from the same logic.
 */
export function interpolateColorStops(stops, t) {
    const clampedT = Math.min(Math.max(t, 0), 1);
    const segmentCount = stops.length - 1;
    const scaled = clampedT * segmentCount;
    const index = Math.min(Math.floor(scaled), segmentCount - 1);
    const localT = scaled - index;

    const stopA = stops[index];
    const stopB = stops[index + 1];

    return [
        Math.round(stopA[0] + (stopB[0] - stopA[0]) * localT),
        Math.round(stopA[1] + (stopB[1] - stopA[1]) * localT),
        Math.round(stopA[2] + (stopB[2] - stopA[2]) * localT),
    ];
}

/**
 * Blends [r,g,b] toward white (amount > 0) or black (amount < 0) by the given fraction.
 * Used to build a highlight/shadow pair from a body's base color for spherical shading.
 */
export function shadeColor([r, g, b], amount) {
    const target = amount > 0 ? 255 : 0;
    const t = Math.min(Math.abs(amount), 1);
    return [
        Math.round(r + (target - r) * t),
        Math.round(g + (target - g) * t),
        Math.round(b + (target - b) * t),
    ];
}

export function getColorForMass(mass) {
    // Sweeps through the full stop sequence (brown -> blue -> green -> red -> yellow ->
    // white) as mass goes from 0 to MAX_MASS, so the heaviest particle turns white.
    const stops = [
        constants.COLORS.BROWN,
        constants.COLORS.BLUE,
        constants.COLORS.GREEN,
        constants.COLORS.RED,
        constants.COLORS.YELLOW,
        constants.COLORS.WHITE,
    ];
    return interpolateColorStops(stops, mass / constants.MAX_MASS);
}

/**
 * The mass gradient exists to show accretion happening - a particle visibly changing
 * color as it merges its way up toward MAX_MASS. With merging off, mass never changes
 * (collide.ts only ever touches position/velocity), so every particle would just sit at
 * whatever color its fixed spawn mass happens to map to - a gradient with nothing left to
 * show. Flattening everyone to white in that mode reflects that mass isn't a visually
 * meaningful, changing quantity here anymore, rather than displaying a gradient that's
 * frozen in place for the entire run.
 */
export function getDisplayColorForMass(mass, mergingEnabled) {
    return mergingEnabled ? getColorForMass(mass) : constants.COLORS.WHITE;
}

/**
 * Heat-style ramp over a 0..1 local-density value (particleSystem.ts's `density` field,
 * populated from collide.ts's candidate count on the CPU paths or webgpuSim.ts's contact
 * count on the GPU path - see constants.ts's DENSITY_BLUR_THRESHOLD): isolated particles
 * read cool blue by default, mid-density regions red, and packed cores climb through
 * yellow into white - so color temperature tracks local crowding instead of a particle's
 * own mass. Stops (and how many of them there are) live in constants.ts's
 * DENSITY_COLOR_STOPS, evenly spaced across 0..1 by the same interpolateColorStops helper
 * the mass gradient above uses - edit that array to retune the palette. The GPU-solver
 * render path (webgpuRenderer.ts's attachSimBuffers) generates its own WGSL copy of this
 * same ramp from those same stops - see that file for why it can only pick up a change on
 * the GPU sim's next (re)build rather than live.
 */
export function densityRamp(t) {
    return interpolateColorStops(constants.DENSITY_COLOR_STOPS, t);
}
