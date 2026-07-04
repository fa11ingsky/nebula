// Static starfield/nebula/galaxy backdrop, rendered once to an offscreen buffer at
// startup (see createNebulaBackground) rather than redrawn every frame.

/**
 * Scatters many small, softly-glowing dabs across the canvas, using a Perlin noise field
 * to decide where "cloud" density is high, so they clump into organic, wispy structures
 * instead of a uniform haze. Dab count scales with canvas area, so density looks the
 * same on a small window and an ultrawide monitor.
 */
function drawNebulaClouds(bg, width, height) {
    const ctx = bg.drawingContext;
    const cloudColors = [
        [138, 43, 226],
        [30, 100, 200],
        [190, 30, 140],
        [40, 160, 170],
        [90, 40, 160],
    ];

    const noiseScale = 0.0035;
    const dabCount = Math.floor((width * height) / 2200);

    for (let i = 0; i < dabCount; i++) {
        const x = bg.random(width);
        const y = bg.random(height);
        const density = bg.noise(x * noiseScale, y * noiseScale);
        if (density < 0.45) {
            continue; // leaves gaps of open space between cloud structures
        }

        const colorField = bg.noise(x * noiseScale, y * noiseScale, 50);
        const [r, g, b] = cloudColors[Math.floor(colorField * cloudColors.length) % cloudColors.length];
        const radius = bg.map(density, 0.45, 1, 12, 60);
        const alpha = bg.map(density, 0.45, 1, 0.025, 0.16);

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

/**
 * Draws one small, tilted smudge meant to read as a spiral galaxy seen from far away: a
 * bright core, a soft elliptical halo (squashed to suggest a viewing angle), and a
 * couple of faint spiral arm traces made of tiny dots along a logarithmic-ish curve.
 */
function drawSpiralGalaxy(bg, x, y, size, angle, tilt, color) {
    const ctx = bg.drawingContext;
    const [r, g, b] = color;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.scale(1, tilt);

    let halo = ctx.createRadialGradient(0, 0, 0, 0, 0, size);
    halo.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.30)`);
    halo.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(0, 0, size, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.55)`;
    for (let arm = 0; arm < 2; arm++) {
        const armOffset = arm * Math.PI;
        for (let t = 0; t < 1; t += 0.04) {
            const theta = t * Math.PI * 2.4 + armOffset;
            const radius = t * size * 0.95;
            const px = Math.cos(theta) * radius;
            const py = Math.sin(theta) * radius;
            const dotSize = (1 - t) * 1.4 + 0.25;
            ctx.beginPath();
            ctx.arc(px, py, dotSize, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    let core = ctx.createRadialGradient(0, 0, 0, 0, 0, size * 0.22);
    core.addColorStop(0, 'rgba(255, 250, 240, 0.9)');
    core.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(0, 0, size * 0.22, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}

/**
 * Scatters a handful of small, distinct spiral galaxies across the background - a
 * separate, sparser layer from the nebula clouds, each individually detailed instead of
 * being another soft blob.
 */
function drawDistantGalaxies(bg, width, height) {
    const palette = [
        [180, 200, 255], // cool blue-white
        [255, 220, 180], // warm gold
        [210, 235, 255],
    ];

    const galaxyCount = Math.floor(bg.random(6, 11));
    for (let i = 0; i < galaxyCount; i++) {
        const x = bg.random(width);
        const y = bg.random(height);
        const size = bg.random(12, 30);
        const angle = bg.random(Math.PI * 2);
        const tilt = bg.random(0.3, 0.7);
        const color = palette[Math.floor(bg.random(palette.length))];

        drawSpiralGalaxy(bg, x, y, size, angle, tilt, color);
    }
}

/**
 * Renders a starfield + layered nebula clouds + a handful of distant spiral galaxies to
 * an offscreen buffer once at startup. The scene is static, so draw() just blits this
 * buffer each frame instead of re-drawing all of this 60 times a second.
 */
export function createNebulaBackground(s, width, height) {
    const bg = s.createGraphics(width, height);
    bg.background(5, 5, 14);
    bg.noiseSeed(Math.floor(bg.random(100000)));

    drawNebulaClouds(bg, width, height);
    drawDistantGalaxies(bg, width, height);

    bg.noStroke();
    const starCount = Math.floor((width * height) / 3200);
    for (let i = 0; i < starCount; i++) {
        bg.fill(255, 255, 255, bg.random(80, 220));
        bg.circle(bg.random(width), bg.random(height), bg.random(0.5, 1.8));
    }

    return bg;
}
