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
