// Cosmetic, per-body surface data (craters/clouds/flares/ring) - only ever read when
// textures are enabled (see particleRender.ts's displayBody).
import constants from '../constants.ts';
import { shadeColor } from './colors.ts';

/**
 * A pure function of mass - regenerated whenever a particle's mass changes (spawn, or
 * absorbing another particle in a merge) so a newly combined body gets its own fresh
 * surface instead of inheriting one twin's exact pattern.
 */
export function generateSurfaceFeatures(mass) {
    // Fewer craters, each nudged into its own ~120-degree sector with a tighter size
    // range, so three independently-placed dimples don't keep landing on top of each
    // other the way five loosely-scattered ones tended to.
    const craterCount = 3;
    const craterSector = (Math.PI * 2) / craterCount;
    const craters = Array.from({ length: craterCount }, (_, i) => ({
        angle: craterSector * i + Math.random() * craterSector * 0.7,
        dist: 0.15 + Math.random() * 0.35,
        size: 0.1 + Math.random() * 0.15,
    }));

    const clouds = Array.from({ length: 4 }, () => ({
        angle: Math.random() * Math.PI * 2,
        dist: Math.random() * 0.5,
        size: 0.4 + Math.random() * 0.4,
    }));

    const flares = Array.from({ length: 3 }, () => ({
        angle: Math.random() * Math.PI * 2,
        dist: Math.random() * 0.6,
        size: 0.15 + Math.random() * 0.2,
        phase: Math.random() * Math.PI * 2,
    }));

    const glowPulsePhase = Math.random() * Math.PI * 2;

    // Only bodies squarely in the "planet" mass range are even eligible, and even then
    // it's a coin flip - "some planets", not all of them.
    const t = Math.min(Math.max(mass / constants.MAX_MASS, 0), 1);
    const eligibleForRing = t > 0.25 && t < 0.8;
    const hasRing = eligibleForRing && Math.random() < 0.35;
    let ring = null;
    if (hasRing) {
        ring = {
            angle: (Math.random() - 0.5) * 0.6, // mostly horizontal, slight tilt variety
            tilt: 0.25 + Math.random() * 0.15, // vertical squash, like a ring seen edge-on-ish
            innerScale: 1.4 + Math.random() * 0.15,
            outerScale: 2.0 + Math.random() * 0.4,
            color: shadeColor([205, 185, 145], (Math.random() - 0.5) * 0.3),
        };
    }

    return { craters, clouds, flares, glowPulsePhase, hasRing, ring };
}
