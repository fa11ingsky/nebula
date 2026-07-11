// Drawing a single body (particle) to the canvas.
import constants from '../constants.ts';
import { shadeColor } from './colors.ts';

/**
 * Draws one half of a ring as an annulus segment (outer arc, then the inner arc traced
 * backwards to cut the hole) in the ring's own tilted/rotated local space. half=0 draws
 * the far/back half, half=1 the near/front half.
 */
export function drawRingHalf(s, x, y, radius, ring, half) {
    const outerRadius = radius * ring.outerScale;
    const innerRadius = radius * ring.innerScale;
    const startAngle = half === 0 ? Math.PI : 0;
    const endAngle = half === 0 ? Math.PI * 2 : Math.PI;

    const ctx = s.drawingContext;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(ring.angle);
    ctx.scale(1, ring.tilt);

    ctx.beginPath();
    ctx.arc(0, 0, outerRadius, startAngle, endAngle);
    ctx.arc(0, 0, innerRadius, endAngle, startAngle, true);
    ctx.closePath();

    const [r, g, b] = ring.color;
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
    ctx.fill();

    ctx.restore();
}

/**
 * Renders one body at (x,y) with the given mass/radius/color/surface-texture data.
 * Parameterized by raw values rather than a particle system reference so it stays
 * decoupled from how particle data happens to be stored.
 *
 * `densityColor`, if given ([r,g,b]), forces the flat-circle fast path with that color
 * regardless of texturesEnabled - the density color mode (colors.ts's densityRamp) shows
 * local crowding, which the textured spherical-shading/craters/clouds path has no
 * meaningful way to layer on top of, so it takes over the whole body's appearance instead.
 */
export function displayBody(s, x, y, mass, radius, r, g, b, colorString, surface, texturesEnabled, densityColor = null) {
    if (densityColor) {
        const ctx = s.drawingContext;
        ctx.fillStyle = `rgb(${densityColor[0]}, ${densityColor[1]}, ${densityColor[2]})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        return;
    }

    if (!texturesEnabled) {
        // Cheap fallback: one flat-colored circle, no extra draw calls per particle - for
        // when the shading/craters/clouds/flares/glow below are costing more than they're
        // worth at high particle counts. Goes straight through the canvas context instead
        // of p5's fill()/ellipse() - at tens of thousands of calls a frame, p5's per-call
        // argument normalization and state-object overhead adds up next to a bare
        // arc()+fill().
        const ctx = s.drawingContext;
        ctx.fillStyle = colorString;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        return;
    }

    s.noStroke();

    // Surface texture gradually shifts with mass: small bodies are cratered asteroids,
    // mid-mass ones pick up soft cloud cover like a planet, and the heaviest ones wash
    // both out under a hot, flaring sun-like glow. Weights overlap deliberately, so the
    // look drifts between categories instead of snapping between them.
    const t = Math.min(Math.max(mass / constants.MAX_MASS, 0), 1);
    const asteroidWeight = Math.max(0, 1 - t / 0.3);
    const cloudWeight = Math.max(0, 1 - Math.abs(t - 0.5) / 0.35);
    const sunWeight = Math.max(0, (t - 0.6) / 0.4);

    // Sun halo, drawn first so it sits behind the body as a glow rather than a ring.
    if (sunWeight > 0.05) {
        const pulse = 1 + 0.08 * Math.sin(s.frameCount * 0.05 + surface.glowPulsePhase);
        const glowRadius = radius * (1.4 + 0.8 * sunWeight) * pulse;
        const ctx = s.drawingContext;
        const gradient = ctx.createRadialGradient(x, y, radius * 0.4, x, y, glowRadius);
        gradient.addColorStop(0, `rgba(255, 235, 150, ${0.55 * sunWeight})`);
        gradient.addColorStop(1, 'rgba(255, 200, 100, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Ring, back half: drawn before the body so it passes behind the far side.
    if (surface.hasRing) {
        drawRingHalf(s, x, y, radius, surface.ring, 0);
    }

    // Base body: a radial gradient offset toward one corner (rather than a flat fill)
    // reads as a lit sphere - bright highlight facing the light, base color across the
    // middle, darkening toward the far limb - instead of a flat painted disc.
    {
        const color = [r, g, b];
        const [hr, hg, hb] = shadeColor(color, 0.55);
        const [dr, dg, db] = shadeColor(color, -0.6);
        const lightOffset = radius * 0.35;
        const ctx = s.drawingContext;
        const sphereGradient = ctx.createRadialGradient(
            x - lightOffset, y - lightOffset, radius * 0.05,
            x, y, radius * 1.05
        );
        sphereGradient.addColorStop(0, `rgb(${hr}, ${hg}, ${hb})`);
        sphereGradient.addColorStop(0.5, `rgb(${r}, ${g}, ${b})`);
        sphereGradient.addColorStop(1, `rgb(${dr}, ${dg}, ${db})`);

        ctx.fillStyle = sphereGradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Asteroid craters: small dark dimples, strongest for the smallest bodies.
    if (asteroidWeight > 0.05) {
        s.fill(0, 0, 0, 100 * asteroidWeight);
        for (const crater of surface.craters) {
            const cx = x + Math.cos(crater.angle) * crater.dist * radius;
            const cy = y + Math.sin(crater.angle) * crater.dist * radius;
            s.ellipse(cx, cy, crater.size * radius);
        }
    }

    // Cloud cover: soft pale patches that peak around mid-mass, like a planet's atmosphere.
    if (cloudWeight > 0.05) {
        s.fill(255, 255, 255, 90 * cloudWeight);
        for (const cloud of surface.clouds) {
            const cx = x + Math.cos(cloud.angle) * cloud.dist * radius;
            const cy = y + Math.sin(cloud.angle) * cloud.dist * radius;
            s.ellipse(cx, cy, cloud.size * radius);
        }
    }

    // Ring, front half: drawn after the body (and clouds) so it passes in front of the near side.
    if (surface.hasRing) {
        drawRingHalf(s, x, y, radius, surface.ring, 1);
    }

    // Surface flares: bright, gently flickering mottling once the body runs hot enough to be sun-like.
    if (sunWeight > 0.05) {
        s.fill(255, 255, 220, 160 * sunWeight);
        for (const flare of surface.flares) {
            const flicker = 0.7 + 0.3 * Math.sin(s.frameCount * 0.08 + flare.phase);
            const cx = x + Math.cos(flare.angle) * flare.dist * radius;
            const cy = y + Math.sin(flare.angle) * flare.dist * radius;
            s.ellipse(cx, cy, flare.size * radius * flicker);
        }
    }
}
