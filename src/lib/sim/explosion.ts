// The short-lived collision flash spawned wherever two particles merge (see merge.ts).
import constants from '../constants.ts';
import { interpolateColorStops } from './colors.ts';

// Mass -> flash color: red -> blue -> yellow -> white, reusing the same named colors as
// the particle mass gradient for visual consistency.
const EXPLOSION_COLOR_STOPS = [
    constants.COLORS.RED,
    constants.COLORS.BLUE,
    constants.COLORS.YELLOW,
    constants.COLORS.WHITE,
];

/**
 * Sized and colored off the combined mass so a small merger is a dim, quick flicker
 * while a big one lights up the screen in intense, hot colors. Each explosion also gets
 * its own random tilt and squash so they don't all read as the same stamped-out circle.
 * Kept as plain objects, not struct-of-arrays: there are far fewer of these than
 * particles, and each is short-lived, so they were never the bottleneck at high particle
 * counts the way the particle system itself was.
 */
export class Explosion {
    constructor(x, y, mass) {
        this.x = x;
        this.y = y;
        this.age = 0;
        this.maxAge = constants.EXPLOSION_DURATION_FRAMES;
        // Same mass -> size relationship as a particle's own radius, just scaled up -
        // so the flash reads as proportional to the size of the bodies that collided.
        this.peakRadius = Math.sqrt(mass) * constants.EXPLOSION_RADIUS_FACTOR;
        this.color = interpolateColorStops(EXPLOSION_COLOR_STOPS, mass / constants.MAX_MASS);

        // Cosmetic-only randomness (never fed back into the physics): gives each flash
        // its own oval shape and orientation instead of always being a perfect circle.
        this.angle = Math.random() * Math.PI * 2;
        this.squashX = 0.55 + Math.random() * 0.45;
        this.squashY = 0.55 + Math.random() * 0.45;
    }

    update() {
        this.age++;
    }

    isDone() {
        return this.age >= this.maxAge;
    }

    display(s) {
        const t = this.age / this.maxAge;
        const growth = 1 - Math.pow(1 - Math.min(t / 0.25, 1), 3); // fast ease-out expansion
        const radius = this.peakRadius * growth;
        const alpha = Math.pow(1 - t, 2); // fades faster than it grows

        if (radius <= 0 || alpha <= 0) {
            return;
        }

        const [r, g, b] = this.color;
        const ctx = s.drawingContext;

        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        ctx.scale(this.squashX, this.squashY);

        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
        gradient.addColorStop(0, `rgba(255, 255, 250, ${0.95 * alpha})`);
        gradient.addColorStop(0.35, `rgba(${r}, ${g}, ${b}, ${0.8 * alpha})`);
        gradient.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, ${0.35 * alpha})`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }
}
