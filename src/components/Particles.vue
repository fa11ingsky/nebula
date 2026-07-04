<template>
    <div class="simulation">
        <div ref="canvasContainer"></div>
        <div class="debug-panel" v-if="debugPanelVisible">
            <div class="debug-title">Center of Mass</div>
            <div>x: {{ centerOfMass.x.toFixed(2) }}</div>
            <div>y: {{ centerOfMass.y.toFixed(2) }}</div>
            <div class="debug-title">Energy</div>
            <div>kinetic:   {{ kineticEnergy.toFixed(2) }}</div>
            <div>potential: {{ potentialEnergy.toFixed(2) }}</div>
            <div>total:     {{ (kineticEnergy + potentialEnergy).toFixed(2) }}</div>
        </div>
        <div class="settings-panel" v-if="settingsOpen">
            <label class="settings-row">
                <input type="checkbox" :checked="centralMassEnabled" @change="toggleCentralMass()" />
                Central Mass
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="debugPanelVisible" />
                Show Debug Info
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="texturesEnabled" />
                Enable Textures
            </label>
            <label class="settings-row">
                <input type="checkbox" v-model="crosshairVisible" />
                Show Crosshair
            </label>

            <div class="settings-group">
                <div class="settings-label">Angular Momentum</div>
                <label class="settings-row" v-for="value in angularMomentumOptions" :key="'am-' + value">
                    <input type="radio" name="angularMomentum" :checked="angularMomentum === value" @change="setAngularMomentum(value)" />
                    {{ value.toLocaleString() }}
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Gravitational Constant</div>
                <label class="settings-row" v-for="value in gravitationalConstantOptions" :key="'g-' + value">
                    <input type="radio" name="gravitationalConstant" :checked="gravitationalConstant === value" @change="setGravitationalConstant(value)" />
                    {{ value }}
                </label>
            </div>

            <div class="settings-group">
                <div class="settings-label">Particle Count</div>
                <label class="settings-row" v-for="value in totalParticlesOptions" :key="'p-' + value">
                    <input type="radio" name="totalParticles" :checked="totalParticles === value" @change="setTotalParticles(value)" />
                    {{ value.toLocaleString() }}
                </label>
            </div>
        </div>
        <div class="controls">
            <button id="stopButton" :class="{ active: stopped }" @click="stopped = !stopped">{{ stopped ? 'Resume' : 'Stop' }}</button>
            <button ref="restartButton" @click="resetSim()">Restart</button>
            <button :class="{ active: settingsOpen }" @click="settingsOpen = !settingsOpen">Settings</button>
        </div>
    </div>
</template>

<script>
    import p5 from 'p5';
    import constants from '../lib/constants.ts';

    /**
     * Linearly interpolates through a sequence of [r,g,b] stops based on t (0..1), picking
     * whichever segment t falls into. Shared by the particle mass gradient and the
     * explosion mass gradient so both read from the same logic.
     */
    function interpolateColorStops(stops, t) {
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
    function shadeColor([r, g, b], amount) {
        const target = amount > 0 ? 255 : 0;
        const t = Math.min(Math.abs(amount), 1);
        return [
            Math.round(r + (target - r) * t),
            Math.round(g + (target - g) * t),
            Math.round(b + (target - b) * t),
        ];
    }

    class Particle {
        constructor(x, y, s, mass = constants.MAX_MASS / constants.TOTAL_PARTICLES) {
            this.position = s.createVector(x, y);
            // Placeholder; initializeAngularMomentum() assigns the real orbital velocity
            // once every particle in the system exists, since it depends on all of them.
            this.velocity = s.createVector(0, 0);
            this.mass = mass;
            this.radius = Math.sqrt(this.mass); // Particle radius
            this.color = this.getColor(this.mass);
            this.acceleration = s.createVector(constants.GRAVITY.X, constants.GRAVITY.Y);
            this.s = s;
            this.generateSurfaceFeatures();

        }

        // Fixed, randomized layout for this body's craters/clouds/flares, stored as offsets
        // (fraction of radius, so they scale automatically as the body grows) rather than
        // redrawn from scratch every frame - a texture that changes shape every frame would
        // just read as noise. Regenerated on merge() so a newly combined body gets its own
        // fresh surface instead of inheriting one twin's exact pattern.
        generateSurfaceFeatures() {
            // Fewer craters, each nudged into its own ~120-degree sector with a tighter size
            // range, so three independently-placed dimples don't keep landing on top of
            // each other the way five loosely-scattered ones tended to.
            const craterCount = 3;
            const craterSector = (Math.PI * 2) / craterCount;
            this.craters = Array.from({ length: craterCount }, (_, i) => ({
                angle: craterSector * i + Math.random() * craterSector * 0.7,
                dist: 0.15 + Math.random() * 0.35,
                size: 0.1 + Math.random() * 0.15,
            }));

            this.clouds = Array.from({ length: 4 }, () => ({
                angle: Math.random() * Math.PI * 2,
                dist: Math.random() * 0.5,
                size: 0.4 + Math.random() * 0.4,
            }));

            this.flares = Array.from({ length: 3 }, () => ({
                angle: Math.random() * Math.PI * 2,
                dist: Math.random() * 0.6,
                size: 0.15 + Math.random() * 0.2,
                phase: Math.random() * Math.PI * 2,
            }));

            this.glowPulsePhase = Math.random() * Math.PI * 2;

            // Only bodies squarely in the "planet" mass range are even eligible, and even
            // then it's a coin flip - "some planets", not all of them.
            const t = Math.min(Math.max(this.mass / constants.MAX_MASS, 0), 1);
            const eligibleForRing = t > 0.25 && t < 0.8;
            this.hasRing = eligibleForRing && Math.random() < 0.35;
            if (this.hasRing) {
                this.ring = {
                    angle: (Math.random() - 0.5) * 0.6, // mostly horizontal, slight tilt variety
                    tilt: 0.25 + Math.random() * 0.15, // vertical squash, like a ring seen edge-on-ish
                    innerScale: 1.4 + Math.random() * 0.15,
                    outerScale: 2.0 + Math.random() * 0.4,
                    color: shadeColor([205, 185, 145], (Math.random() - 0.5) * 0.3),
                };
            }
        }

        // Draws one half of the ring as an annulus segment (outer arc, then the inner arc
        // traced backwards to cut the hole) in the ring's own tilted/rotated local space.
        // half=0 draws the far/back half, half=1 the near/front half.
        drawRingHalf(half) {
            const ctx = this.s.drawingContext;
            const outerRadius = this.radius * this.ring.outerScale;
            const innerRadius = this.radius * this.ring.innerScale;
            const startAngle = half === 0 ? Math.PI : 0;
            const endAngle = half === 0 ? Math.PI * 2 : Math.PI;

            ctx.save();
            ctx.translate(this.position.x, this.position.y);
            ctx.rotate(this.ring.angle);
            ctx.scale(1, this.ring.tilt);

            ctx.beginPath();
            ctx.arc(0, 0, outerRadius, startAngle, endAngle);
            ctx.arc(0, 0, innerRadius, endAngle, startAngle, true);
            ctx.closePath();

            const [r, g, b] = this.ring.color;
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
            ctx.fill();

            ctx.restore();
        }

        resetAcceleration() {
            // Reset to the base external field; pairwise gravity accumulates on top of this each frame
            this.acceleration.set(constants.GRAVITY.X, constants.GRAVITY.Y);
        }

        // Applies half of this frame's acceleration to velocity. Called twice per frame -
        // once with last frame's acceleration before the position moves, once with the
        // freshly recomputed acceleration after - which is the "kick-drift-kick" leapfrog
        // scheme. It costs the same single gravity evaluation per frame as plain Euler
        // integration, but is far more stable/realistic for orbital motion.
        kick(dt) {
            this.velocity.x += this.acceleration.x * dt;
            this.velocity.y += this.acceleration.y * dt;
        }

        drift() {
            // No boundary: with nothing external ever acting on a particle, gravity and
            // merges (both exactly momentum-conserving) are the only things that can ever
            // change total system momentum - so it stays exactly constant for the whole run.
            this.position.add(this.velocity);
        }

        merge(other) {
            // Combine particles based on momentum conservation
            let totalMass = this.mass + other.mass;
            let p1 = this.velocity.mult(this.mass);
            let p2 = other.velocity.mult(other.mass);
            let p_t = p1.add(p2);
            this.velocity = p_t.div(totalMass);
            // Weight by mass so the merged body lands at the true center of mass instead
            // of the geometric midpoint - otherwise merging unequal masses would inject
            // spurious angular momentum into the system.
            this.position = this.position.mult(this.mass).add(other.position.mult(other.mass)).div(totalMass);
            this.radius = Math.sqrt(totalMass); // Adjust radius based on combined mass
            //console.log(`Merge: ${this.radius}`)
            this.mass = totalMass
            this.color = this.getColor(this.mass)
            this.generateSurfaceFeatures();
        }

        display(texturesEnabled) {
            this.s.noStroke();

            if (!texturesEnabled) {
                // Cheap fallback: one flat-colored circle, no extra draw calls per particle -
                // for when the shading/craters/clouds/flares/glow below are costing more than
                // they're worth at high particle counts.
                this.s.fill(this.color[0], this.color[1], this.color[2]);
                this.s.ellipse(this.position.x, this.position.y, this.radius * 2);
                return;
            }

            // Surface texture gradually shifts with mass: small bodies are cratered
            // asteroids, mid-mass ones pick up soft cloud cover like a planet, and the
            // heaviest ones wash both out under a hot, flaring sun-like glow. Weights
            // overlap deliberately, so the look drifts between categories instead of
            // snapping between them.
            const t = Math.min(Math.max(this.mass / constants.MAX_MASS, 0), 1);
            const asteroidWeight = Math.max(0, 1 - t / 0.3);
            const cloudWeight = Math.max(0, 1 - Math.abs(t - 0.5) / 0.35);
            const sunWeight = Math.max(0, (t - 0.6) / 0.4);

            // Sun halo, drawn first so it sits behind the body as a glow rather than a ring.
            if (sunWeight > 0.05) {
                const pulse = 1 + 0.08 * Math.sin(this.s.frameCount * 0.05 + this.glowPulsePhase);
                const glowRadius = this.radius * (1.4 + 0.8 * sunWeight) * pulse;
                const ctx = this.s.drawingContext;
                const gradient = ctx.createRadialGradient(
                    this.position.x, this.position.y, this.radius * 0.4,
                    this.position.x, this.position.y, glowRadius
                );
                gradient.addColorStop(0, `rgba(255, 235, 150, ${0.55 * sunWeight})`);
                gradient.addColorStop(1, 'rgba(255, 200, 100, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(this.position.x, this.position.y, glowRadius, 0, Math.PI * 2);
                ctx.fill();
            }

            // Ring, back half: drawn before the body so it passes behind the far side.
            if (this.hasRing) {
                this.drawRingHalf(0);
            }

            // Base body: a radial gradient offset toward one corner (rather than a flat fill)
            // reads as a lit sphere - bright highlight facing the light, base color across
            // the middle, darkening toward the far limb - instead of a flat painted disc.
            {
                const [hr, hg, hb] = shadeColor(this.color, 0.55);
                const [dr, dg, db] = shadeColor(this.color, -0.6);
                const lightOffset = this.radius * 0.35;
                const ctx = this.s.drawingContext;
                const sphereGradient = ctx.createRadialGradient(
                    this.position.x - lightOffset, this.position.y - lightOffset, this.radius * 0.05,
                    this.position.x, this.position.y, this.radius * 1.05
                );
                sphereGradient.addColorStop(0, `rgb(${hr}, ${hg}, ${hb})`);
                sphereGradient.addColorStop(0.5, `rgb(${this.color[0]}, ${this.color[1]}, ${this.color[2]})`);
                sphereGradient.addColorStop(1, `rgb(${dr}, ${dg}, ${db})`);

                ctx.fillStyle = sphereGradient;
                ctx.beginPath();
                ctx.arc(this.position.x, this.position.y, this.radius, 0, Math.PI * 2);
                ctx.fill();
            }

            // Asteroid craters: small dark dimples, strongest for the smallest bodies.
            if (asteroidWeight > 0.05) {
                this.s.fill(0, 0, 0, 100 * asteroidWeight);
                for (let crater of this.craters) {
                    const cx = this.position.x + Math.cos(crater.angle) * crater.dist * this.radius;
                    const cy = this.position.y + Math.sin(crater.angle) * crater.dist * this.radius;
                    this.s.ellipse(cx, cy, crater.size * this.radius);
                }
            }

            // Cloud cover: soft pale patches that peak around mid-mass, like a planet's atmosphere.
            if (cloudWeight > 0.05) {
                this.s.fill(255, 255, 255, 90 * cloudWeight);
                for (let cloud of this.clouds) {
                    const cx = this.position.x + Math.cos(cloud.angle) * cloud.dist * this.radius;
                    const cy = this.position.y + Math.sin(cloud.angle) * cloud.dist * this.radius;
                    this.s.ellipse(cx, cy, cloud.size * this.radius);
                }
            }

            // Ring, front half: drawn after the body (and clouds) so it passes in front of the near side.
            if (this.hasRing) {
                this.drawRingHalf(1);
            }

            // Surface flares: bright, gently flickering mottling once the body runs hot enough to be sun-like.
            if (sunWeight > 0.05) {
                this.s.fill(255, 255, 220, 160 * sunWeight);
                for (let flare of this.flares) {
                    const flicker = 0.7 + 0.3 * Math.sin(this.s.frameCount * 0.08 + flare.phase);
                    const cx = this.position.x + Math.cos(flare.angle) * flare.dist * this.radius;
                    const cy = this.position.y + Math.sin(flare.angle) * flare.dist * this.radius;
                    this.s.ellipse(cx, cy, flare.size * this.radius * flicker);
                }
            }
        }

        getColor(mass) {
            // Sweeps through the full stop sequence (brown -> blue -> green -> red -> yellow
            // -> white) as mass goes from 0 to MAX_MASS, so the heaviest particle turns white.
            // Returned as an [r,g,b] array (rather than a CSS string) so display() can shade
            // it into a highlight/shadow pair for the spherical lighting effect.
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
    }// end Particle class

    // Mass -> flash color: red -> blue -> yellow -> white, reusing the same named colors
    // as the particle mass gradient for visual consistency.
    const EXPLOSION_COLOR_STOPS = [
        constants.COLORS.RED,
        constants.COLORS.BLUE,
        constants.COLORS.YELLOW,
        constants.COLORS.WHITE,
    ];

    /**
     * A short-lived collision flash spawned wherever two particles merge, sized and colored
     * off the combined mass so a small merger is a dim, quick flicker while a big one lights
     * up the screen in intense, hot colors. Each explosion also gets its own random tilt and
     * squash so they don't all read as the same stamped-out circle.
     */
    class Explosion {
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
    } // end Explosion class

    /**
     * A Barnes-Hut quadtree node: a square region of space that either holds a single
     * particle directly (or, past QUADTREE_MAX_DEPTH, a small flat bucket of coincident-ish
     * particles), or has been subdivided into four children. Every node - leaf or internal -
     * tracks the total mass and center of mass of everything inside it, accumulated
     * incrementally as particles are inserted, which is what lets a distant node be treated
     * as a single aggregate body during force/range queries instead of visiting its contents
     * one by one.
     */
    class QuadTreeNode {
        constructor(x, y, size) {
            this.x = x;
            this.y = y;
            this.size = size;
            this.occupant = null;
            this.bucket = null;
            this.children = null;
            this.mass = 0;
            this.comX = 0;
            this.comY = 0;
        }

        insert(particle, depth) {
            // Every node on the insertion path folds this particle into its running
            // mass-weighted center of mass, regardless of whether it ends up a leaf or
            // internal node - this is what makes "treat this whole subtree as one point
            // mass" possible later.
            if (this.mass === 0) {
                this.comX = particle.position.x;
                this.comY = particle.position.y;
            } else {
                const newMass = this.mass + particle.mass;
                this.comX = (this.comX * this.mass + particle.position.x * particle.mass) / newMass;
                this.comY = (this.comY * this.mass + particle.position.y * particle.mass) / newMass;
            }
            this.mass += particle.mass;

            if (this.children) {
                this.childFor(particle).insert(particle, depth + 1);
                return;
            }

            if (this.bucket) {
                this.bucket.push(particle);
                return;
            }

            if (this.occupant === null) {
                this.occupant = particle;
                return;
            }

            if (depth >= constants.QUADTREE_MAX_DEPTH) {
                // Particles landed on (almost) the same point and there's no room left to
                // subdivide meaningfully - fall back to a flat list instead of recursing forever.
                this.bucket = [this.occupant, particle];
                this.occupant = null;
                return;
            }

            this.subdivide();
            const existing = this.occupant;
            this.occupant = null;
            this.childFor(existing).insert(existing, depth + 1);
            this.childFor(particle).insert(particle, depth + 1);
        }

        subdivide() {
            const half = this.size / 2;
            this.children = [
                new QuadTreeNode(this.x, this.y, half),              // NW
                new QuadTreeNode(this.x + half, this.y, half),        // NE
                new QuadTreeNode(this.x, this.y + half, half),        // SW
                new QuadTreeNode(this.x + half, this.y + half, half), // SE
            ];
        }

        // Uses the particle's clamped _treeX/_treeY (see buildQuadtree) rather than its true
        // position - that's what keeps a single far-flung outlier from forcing the whole
        // tree (and every normal, tightly-packed particle in it) through extra levels of
        // subdivision just to accommodate one point way out on its own.
        childFor(particle) {
            const half = this.size / 2;
            const east = particle._treeX >= this.x + half ? 1 : 0;
            const south = particle._treeY >= this.y + half ? 1 : 0;
            return this.children[south * 2 + east];
        }
    } // end QuadTreeNode class

    /**
     * Builds a fresh quadtree covering the swarm. Rebuilt from scratch each time it's needed
     * (positions change every frame) rather than incrementally updated - at O(n log n) this
     * is cheap next to the O(n^2) it's replacing.
     *
     * Sized off the RMS spread from the center of mass (times a generous safety margin)
     * rather than the exact min/max extent of every particle. A single particle flung far
     * away by a slingshot would otherwise balloon a min/max box to its position, forcing
     * every other, tightly-packed particle through many extra levels of subdivision just to
     * separate out normally - degrading both performance and, past QUADTREE_MAX_DEPTH,
     * correctness (falling back to oversized flat buckets that treat distinguishable
     * particles as an undifferentiated pile). Particles further out than that are clamped to
     * the tree's boundary for structural purposes only (see childFor) - their true position
     * and mass still feed the aggregate mass/center-of-mass math untouched, so the physics
     * stays correct; only which quadrant a stray particle nominally sorts into is affected,
     * and at that distance its exact position barely matters to anyone's force anyway.
     */
    function buildQuadtree(particles) {
        const com = computeCenterOfMass(particles);

        let sumSqDeviation = 0;
        for (const particle of particles) {
            const dx = particle.position.x - com.x;
            const dy = particle.position.y - com.y;
            sumSqDeviation += dx * dx + dy * dy;
        }
        const rmsSpread = particles.length > 0 ? Math.sqrt(sumSqDeviation / particles.length) : 1;

        const halfSize = Math.max(rmsSpread * 6, 10);
        const size = halfSize * 2;
        const rootX = com.x - halfSize;
        const rootY = com.y - halfSize;
        const maxCoord = rootX + size * (1 - 1e-9); // just inside the far boundary

        for (const particle of particles) {
            particle._treeX = Math.min(Math.max(particle.position.x, rootX), maxCoord);
            particle._treeY = Math.min(Math.max(particle.position.y, rootY), maxCoord);
        }

        const root = new QuadTreeNode(rootX, rootY, size);
        for (const particle of particles) {
            root.insert(particle, 0);
        }
        return root;
    }

    /**
     * Accumulates gravitational acceleration onto `particle` from one specific other body,
     * softened over their combined radius (see GRAVITY_SOFTENING_FACTOR) so the force stays
     * finite even at zero separation. Only updates `particle`'s own acceleration - see
     * applyTreeGravity for why that's still momentum-conserving overall.
     */
    function applyDirectGravity(particle, other, dx, dy, distSq) {
        const combinedRadius = particle.radius + other.radius;
        const softenedDistSq = distSq + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
        const dist = Math.sqrt(softenedDistSq);
        const scalar = (constants.GRAVITATIONAL_CONSTANT * other.mass) / (softenedDistSq * dist);
        particle.acceleration.x += dx * scalar;
        particle.acceleration.y += dy * scalar;
    }

    /**
     * Same as applyDirectGravity, but for treating an entire distant subtree as a single
     * point mass at its center of mass. There's no specific "other body" to size a softening
     * length from here, but the opening-angle test in applyTreeGravity already guarantees
     * this is only used when the node is comfortably far away, so a minimal softening off
     * just this particle's own radius is enough of a safety net.
     */
    function applyAggregateGravity(particle, otherMass, dx, dy, distSq) {
        const softenedDistSq = distSq + particle.radius * particle.radius;
        const dist = Math.sqrt(softenedDistSq);
        const scalar = (constants.GRAVITATIONAL_CONSTANT * otherMass) / (softenedDistSq * dist);
        particle.acceleration.x += dx * scalar;
        particle.acceleration.y += dy * scalar;
    }

    /**
     * Accumulates the total gravitational acceleration on `particle` from everywhere else in
     * the tree (Barnes-Hut). For a node that's far enough away relative to its size (the
     * opening-angle test), the whole subtree is treated as one point mass instead of
     * descending into it - O(log n) node visits per particle instead of O(n).
     *
     * On momentum conservation: every particle runs this traversal independently, and only
     * ever updates its OWN acceleration (never anyone else's). For two individual particles
     * resolved down to actual leaves, both sides independently compute the same physics from
     * their own perspective, which is exactly equal-and-opposite by construction - so
     * near-field interactions stay (numerically, near-exactly) momentum-conserving, the same
     * as the old direct pairwise sum. Far-field interactions approximated as a cluster are
     * the exception: only the querying particle gets a force from "the cluster," with no
     * single reaction applied back to the many individual bodies inside it. That's the real
     * accuracy/speed trade this algorithm makes - total momentum is very close to conserved,
     * not exact to the bit, in exchange for going from O(n^2) to O(n log n).
     */
    function applyTreeGravity(particle, node) {
        if (node.mass === 0) {
            return;
        }

        const dx = node.comX - particle.position.x;
        const dy = node.comY - particle.position.y;
        const distSq = dx * dx + dy * dy;

        if (!node.children) {
            if (node.occupant !== null) {
                if (node.occupant !== particle) {
                    applyDirectGravity(particle, node.occupant, dx, dy, distSq);
                }
            } else if (node.bucket) {
                for (const other of node.bucket) {
                    if (other === particle) continue;
                    const odx = other.position.x - particle.position.x;
                    const ody = other.position.y - particle.position.y;
                    applyDirectGravity(particle, other, odx, ody, odx * odx + ody * ody);
                }
            }
            return;
        }

        // Opening-angle test: (size/distance)^2 < theta^2 -> approximate; otherwise descend.
        if (distSq > 0 && (node.size * node.size) / distSq < constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA) {
            applyAggregateGravity(particle, node.mass, dx, dy, distSq);
            return;
        }

        for (const child of node.children) {
            applyTreeGravity(particle, child);
        }
    }

    /**
     * Applies gravity to every particle, building a tree first unless one is already
     * provided. Call after resetAcceleration() and before the second half-kick.
     *
     * Accepting a pre-built tree lets the caller skip a second full rebuild on frames where
     * nothing merged: positions haven't changed since the merge-detection tree was built, so
     * that tree is already exactly correct for gravity too. Rebuilding it anyway would be
     * pure waste - and became a bigger one once merges got rarer (see mergeParticles' notes
     * on MIN_MERGE_VELOCITY), since more frames now have nothing merge on them at all.
     */
    function computeGravity(particles, tree) {
        const gravityTree = tree || buildQuadtree(particles);
        for (const particle of particles) {
            applyTreeGravity(particle, gravityTree);
        }
        return gravityTree;
    }

    /**
     * Total kinetic energy of the system: sum(0.5 * m * v^2). Exact, O(n) - no approximation
     * needed here, unlike potential energy below.
     */
    function computeKineticEnergy(particles) {
        let ke = 0;
        for (const particle of particles) {
            const speedSq = particle.velocity.x * particle.velocity.x + particle.velocity.y * particle.velocity.y;
            ke += 0.5 * particle.mass * speedSq;
        }
        return ke;
    }

    /**
     * Accumulates the gravitational potential energy between `particle` and everything else
     * in the tree, using the exact same direct/aggregate split (and opening-angle criterion)
     * as applyTreeGravity - so this is a Barnes-Hut approximation of PE, not an exact sum,
     * consistent with the fact that the forces driving this simulation are approximated the
     * same way. Softened the same way as force (see applyDirectGravity/applyAggregateGravity)
     * so a pair well inside the merge threshold doesn't produce a singular energy value.
     */
    function accumulatePotentialEnergy(particle, node) {
        if (node.mass === 0) {
            return 0;
        }

        const dx = node.comX - particle.position.x;
        const dy = node.comY - particle.position.y;
        const distSq = dx * dx + dy * dy;

        if (!node.children) {
            if (node.occupant !== null) {
                if (node.occupant === particle) {
                    return 0;
                }
                const combinedRadius = particle.radius + node.occupant.radius;
                const softenedDistSq = distSq + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
                return -(constants.GRAVITATIONAL_CONSTANT * particle.mass * node.occupant.mass) / Math.sqrt(softenedDistSq);
            }
            if (node.bucket) {
                let pe = 0;
                for (const other of node.bucket) {
                    if (other === particle) continue;
                    const odx = other.position.x - particle.position.x;
                    const ody = other.position.y - particle.position.y;
                    const combinedRadius = particle.radius + other.radius;
                    const softenedDistSq = odx * odx + ody * ody + combinedRadius * combinedRadius * constants.GRAVITY_SOFTENING_FACTOR;
                    pe += -(constants.GRAVITATIONAL_CONSTANT * particle.mass * other.mass) / Math.sqrt(softenedDistSq);
                }
                return pe;
            }
            return 0;
        }

        if (distSq > 0 && (node.size * node.size) / distSq < constants.BARNES_HUT_THETA * constants.BARNES_HUT_THETA) {
            const softenedDistSq = distSq + particle.radius * particle.radius;
            return -(constants.GRAVITATIONAL_CONSTANT * particle.mass * node.mass) / Math.sqrt(softenedDistSq);
        }

        let pe = 0;
        for (const child of node.children) {
            pe += accumulatePotentialEnergy(particle, child);
        }
        return pe;
    }

    /**
     * Total gravitational potential energy of the system, reusing the same tree already
     * built for gravity this frame rather than a third rebuild. Every particle traverses the
     * tree independently (same as applyTreeGravity), so each real pair's energy gets counted
     * once from each side - hence the final /2.
     */
    function computePotentialEnergy(particles, tree) {
        let pe = 0;
        for (const particle of particles) {
            pe += accumulatePotentialEnergy(particle, tree);
        }
        return pe / 2;
    }

    /**
     * Collects every particle within searchRadius of `particle` into `out`, pruning subtrees
     * whose bounding box can't possibly contain a point that close (closest-point-on-box test).
     */
    function findMergeCandidates(node, particle, searchRadius, out) {
        if (node.mass === 0) {
            return;
        }

        const closestX = Math.max(node.x, Math.min(particle.position.x, node.x + node.size));
        const closestY = Math.max(node.y, Math.min(particle.position.y, node.y + node.size));
        const dx = particle.position.x - closestX;
        const dy = particle.position.y - closestY;
        if (dx * dx + dy * dy > searchRadius * searchRadius) {
            return;
        }

        if (node.children) {
            for (const child of node.children) {
                findMergeCandidates(child, particle, searchRadius, out);
            }
            return;
        }

        if (node.occupant !== null) {
            if (node.occupant !== particle) {
                out.push(node.occupant);
            }
        } else if (node.bucket) {
            for (const other of node.bucket) {
                if (other !== particle) out.push(other);
            }
        }
    }

    /**
     * Checks for merges via a quadtree range query instead of an all-pairs scan, returning
     * the surviving particles. Every merge appends an Explosion to `explosions`, sized off
     * the merged body's combined mass. Pure merge detection only - gravity is applied
     * separately (computeGravity).
     * @param {Array} particles - array of initial particles
     * @param {Array} explosions - collision flashes, mutated in place
     * @returns {{particles: Array, tree: QuadTreeNode, anyMerged: boolean}} - the surviving
     *   particles, the tree built to find them (reusable for gravity only if anyMerged is
     *   false - a merge mutates the survivor's position/mass, which would make this tree
     *   stale for it), and whether anything actually merged this frame.
     */
    function mergeParticles(particles, explosions) {
        const tree = buildQuadtree(particles);

        // Upper bound on how far apart two particles could possibly be and still merge
        // (mergeThreshold = (radius_a + radius_b)/2), so the range query never misses a
        // legitimate candidate no matter which other particle it turns out to be.
        let globalMaxRadius = 0;
        for (const particle of particles) {
            if (particle.radius > globalMaxRadius) globalMaxRadius = particle.radius;
        }

        const candidates = [];

        for (const particle of particles) {
            if (particle._removed) {
                continue;
            }

            candidates.length = 0;
            findMergeCandidates(tree, particle, (particle.radius + globalMaxRadius) / 2, candidates);

            for (const other of candidates) {
                if (other._removed) continue;

                const dx = other.position.x - particle.position.x;
                const dy = other.position.y - particle.position.y;
                const distSq = dx * dx + dy * dy;
                const minDistance = (particle.radius + other.radius) / 2;

                if (distSq > minDistance * minDistance) {
                    continue; // not touching at all
                }

                // Touching isn't enough on its own - a slow graze (e.g. two bodies briefly
                // in contact while drifting past each other in similar orbits) just passes
                // through instead of fusing. Only a close approach with enough relative
                // (closing) speed actually merges.
                const relVx = other.velocity.x - particle.velocity.x;
                const relVy = other.velocity.y - particle.velocity.y;
                const relSpeedSq = relVx * relVx + relVy * relVy;

                if (relSpeedSq < constants.MIN_MERGE_VELOCITY * constants.MIN_MERGE_VELOCITY) {
                    continue; // touching, but too gentle to fuse
                }

                particle.merge(other);
                other._removed = true;
                explosions.push(new Explosion(particle.position.x, particle.position.y, particle.mass));
                break; // one merge per particle per frame, same as before
            }
        }

        // Filtered AFTER every particle has had its turn, not committed incrementally
        // during the loop above: a particle that finds no match on its own turn can still
        // be absorbed later by a *different* particle's turn in the same pass (merge
        // candidates come from spatial proximity, not array order, so a particle with two
        // valid partners only merges with whichever is checked first - the other partner
        // can still come back for it on its own turn). Committing early left "ghosts": a
        // particle counted once on its own and again as part of whoever absorbed it later,
        // duplicating mass out of nowhere every time it happened.
        const survivors = particles.filter((particle) => !particle._removed);
        return { particles: survivors, tree, anyMerged: survivors.length !== particles.length };
    } // end of mergeParticles

    /**
     * Picks a uniformly random point within radius of center, purely to shape the initial
     * cluster - there's no boundary/wall tied to this anymore.
     */
    function randomSpawnPoint(s, center, radius) {
        let angle = s.random(Math.PI * 2);
        let r = radius * Math.sqrt(s.random(1));
        return s.createVector(center.x + r * Math.cos(angle), center.y + r * Math.sin(angle));
    }

    /**
     * Populates `particles` with the swarm, optionally followed by a heavy central body,
     * then sets up the system's initial rotation. Shared by both the first load and Restart
     * so the two never drift out of sync with each other. Returns the world-space point
     * spawning was centered on, so callers can keep a fixed reference for measuring drift
     * later - independent of wherever the canvas happens to be sized to at the time.
     */
    function spawnParticles(s, particles, includeCentralMass) {
        const center = s.createVector(s.width / 2, s.height / 2);
        const spawnRadius = Math.min(s.width, s.height) / 2 * constants.SPAWN_RADIUS_FRACTION;

        for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
            let spawn = randomSpawnPoint(s, center, spawnRadius);
            particles.push(new Particle(spawn.x, spawn.y, s));
        }

        if (includeCentralMass) {
            const centralMass = constants.MAX_MASS * constants.CENTRAL_MASS_FRACTION;
            particles.push(new Particle(center.x, center.y, s, centralMass));
        }

        initializeAngularMomentum(particles, constants.TOTAL_ANGULAR_MOMENTUM);

        return { x: center.x, y: center.y };
    }

    /**
     * Mass-weighted center of the given particles. If total momentum is exactly zero (as
     * initializeAngularMomentum sets up below), this point never moves - regardless of how
     * far individual particles wander from it.
     */
    function computeCenterOfMass(particles) {
        let totalMass = 0;
        let x = 0;
        let y = 0;
        for (let particle of particles) {
            totalMass += particle.mass;
            x += particle.mass * particle.position.x;
            y += particle.mass * particle.position.y;
        }
        return { x: x / totalMass, y: y / totalMass };
    }

    /**
     * Sets every particle's initial velocity to a rigid-body-style rotation about the
     * system's own center of mass (velocity proportional to distance from the center of
     * mass, all sharing one angular velocity omega), which guarantees two things at once:
     *  - total linear momentum is exactly zero: sum(m_i * v_i) = omega x sum(m_i * r_i'),
     *    and mass-weighted offsets from the center of mass always sum to zero by definition.
     *  - total angular momentum about that center of mass - and, since linear momentum is
     *    zero, about any other fixed point too - equals exactly targetL, via the standard
     *    rigid-body relation L = I * omega.
     */
    function initializeAngularMomentum(particles, targetL) {
        const com = computeCenterOfMass(particles);

        let momentOfInertia = 0;
        let offsets = particles.map((particle) => {
            let dx = particle.position.x - com.x;
            let dy = particle.position.y - com.y;
            momentOfInertia += particle.mass * (dx * dx + dy * dy);
            return { dx, dy };
        });

        if (momentOfInertia < 1e-6) {
            return; // every particle sits at the center of mass; no rotation is meaningful
        }

        const omega = targetL / momentOfInertia;

        particles.forEach((particle, i) => {
            let { dx, dy } = offsets[i];
            particle.velocity.set(-omega * dy, omega * dx);
        });
    }

    /**
     * Scatters many small, softly-glowing dabs across the canvas, using a Perlin noise
     * field to decide where "cloud" density is high, so they clump into organic, wispy
     * structures instead of a uniform haze. Dab count scales with canvas area, so density
     * looks the same on a small window and an ultrawide monitor.
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
     * Draws one small, tilted smudge meant to read as a spiral galaxy seen from far away:
     * a bright core, a soft elliptical halo (squashed to suggest a viewing angle), and a
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
     * separate, sparser layer from the nebula clouds, each individually detailed instead
     * of being another soft blob.
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
    function createNebulaBackground(s, width, height) {
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

    export default {
        data() {
            return {
                stopped: false,
                centralMassEnabled: false,
                debugPanelVisible: false,
                settingsOpen: false,
                // Lets the surface-texture rendering (spherical shading/craters/clouds/flares/glow)
                // be switched off to fall back to plain flat circles when it's costing too
                // much at high particle counts. Off by default for that reason.
                texturesEnabled: false,
                crosshairVisible: false,
                // Mirror constants.ts's current values as the selected radio option, so the
                // settings panel opens already showing what the sim actually started with.
                angularMomentum: constants.TOTAL_ANGULAR_MOMENTUM,
                gravitationalConstant: constants.GRAVITATIONAL_CONSTANT,
                totalParticles: constants.TOTAL_PARTICLES,
                // Just the option lists for the settings panel's radio groups - defined in
                // constants.ts so adding/removing choices doesn't need a template change.
                // Deliberately exposing only these small arrays, not the whole `constants`
                // object: that object is read directly by the O(n^2) gravity loop every
                // frame, and making it reactive would route every one of those reads through
                // a Vue proxy trap for no benefit.
                angularMomentumOptions: constants.ANGULAR_MOMENTUM_OPTIONS,
                gravitationalConstantOptions: constants.GRAVITATIONAL_CONSTANT_OPTIONS,
                totalParticlesOptions: constants.TOTAL_PARTICLES_OPTIONS,
                // The only per-frame simulation state that's actually reactive - it drives a
                // small text readout in the debug panel, so it has to be visible to the
                // template. Updated only while the panel is open (see sketch.draw()) to
                // avoid triggering a Vue re-render 60 times a second for no reason.
                centerOfMass: { x: 0, y: 0 },
                // Mergers are perfectly inelastic, so unlike momentum/angular momentum this
                // is expected to visibly drop at each collision (radiated away as the "heat"
                // the explosion effect represents) rather than stay constant - that's the
                // point of tracking it.
                kineticEnergy: 0,
                potentialEnergy: 0
            };
        },

        created() {
            // Deliberately kept off Vue's reactivity system: particles are mutated many
            // times each, per frame (position/velocity/acceleration during the O(n^2)
            // gravity pass), and canvas is a large third-party (p5) object. Declaring
            // either in data() would make Vue deep-proxy them, routing every single field
            // access during the hot loop through reactive dependency-tracking for no
            // benefit - nothing in the template ever reads this data; the canvas is drawn
            // imperatively by p5, not through Vue's renderer.
            this.canvas = null;
            this.particles = [];
            this.explosions = [];
            // Fixed world-space point spawning was centered on - used only to measure
            // center-of-mass drift for the debug panel. Deliberately NOT re-derived from the
            // canvas's current width/height on every frame: that's the camera's job, and
            // resizing the canvas would otherwise silently redefine "center" for this
            // measurement without any particle actually having moved.
            this.worldCenter = { x: 0, y: 0 };
        },

        mounted() {
            this.createCanvas();
        },
        methods: {
            resetSim() {
                this.canvas.clear();
                this.particles = [];
                this.explosions = [];
                this.worldCenter = spawnParticles(this.canvas, this.particles, this.centralMassEnabled);
            },
            toggleCentralMass() {
                this.centralMassEnabled = !this.centralMassEnabled;
                this.resetSim();
            },
            setAngularMomentum(value) {
                this.angularMomentum = value;
                // Only consumed at spawn time (initializeAngularMomentum), so it needs a
                // restart to actually take effect.
                constants.TOTAL_ANGULAR_MOMENTUM = value;
                this.resetSim();
            },
            setGravitationalConstant(value) {
                this.gravitationalConstant = value;
                // computeGravity's force functions read this fresh every frame, so it takes
                // effect immediately - no restart needed, unlike the other two settings here.
                constants.GRAVITATIONAL_CONSTANT = value;
            },
            setTotalParticles(value) {
                this.totalParticles = value;
                constants.TOTAL_PARTICLES = value;
                this.resetSim();
            },
            createCanvas() {
                this.canvas = new p5((sketch) => {
                    let nebulaBackground;

                    sketch.setup = () => {
                        //sketch.frameRate(10);
                        // Plain 2D (P2D) renderer: nothing in this sim uses the z-axis, and
                        // many small filled circles are considerably cheaper to draw in P2D
                        // than through WEBGL's per-draw-call material/shader overhead.
                        // Sized to the full viewport rather than leaving a margin.
                        sketch.createCanvas(sketch.windowWidth, sketch.windowHeight).parent(this.$refs.canvasContainer);

                        nebulaBackground = createNebulaBackground(sketch, sketch.width, sketch.height);

                        this.worldCenter = spawnParticles(sketch, this.particles, this.centralMassEnabled);
                    };

                    sketch.windowResized = () => {
                        sketch.resizeCanvas(sketch.windowWidth, sketch.windowHeight);
                        nebulaBackground = createNebulaBackground(sketch, sketch.width, sketch.height);
                    };

                    sketch.draw = () => {
                        if (!this.stopped) {
                            sketch.image(nebulaBackground, 0, 0);

                            // Leapfrog ("kick-drift-kick") integration: half-kick with the
                            // acceleration already sitting on each particle from the end of
                            // last frame, drift positions, recompute gravity at the new
                            // positions, then apply the second half-kick. Same one gravity
                            // evaluation per frame as plain Euler, far better energy behavior
                            // for orbital motion.
                            for (let particle of this.particles) {
                                particle.kick(0.5);
                            }

                            for (let particle of this.particles) {
                                particle.drift();
                            }

                            // Merges use their own quadtree built from pre-merge positions,
                            // and don't touch acceleration at all - gravity is applied
                            // separately, right after. On the (increasingly common, since
                            // MIN_MERGE_VELOCITY filters out most touches) frames where
                            // nothing actually merged, the merge-phase tree is still exactly
                            // correct for gravity too - positions didn't change - so it's
                            // reused instead of paying for a second full rebuild.
                            const mergeResult = mergeParticles(this.particles, this.explosions);
                            this.particles = mergeResult.particles;

                            for (let particle of this.particles) {
                                particle.resetAcceleration();
                            }
                            const gravityTree = computeGravity(this.particles, mergeResult.anyMerged ? null : mergeResult.tree);

                            for (let particle of this.particles) {
                                particle.kick(0.5);
                            }

                            if (this.debugPanelVisible) {
                                // Reuses the tree gravity already built this frame rather than
                                // a third rebuild just for this readout.
                                this.kineticEnergy = computeKineticEnergy(this.particles);
                                this.potentialEnergy = computePotentialEnergy(this.particles, gravityTree);
                            }

                            for (let explosion of this.explosions) {
                                explosion.update();
                            }
                            this.explosions = this.explosions.filter((explosion) => !explosion.isDone());

                            // Camera follows the center of mass, which - with total momentum
                            // held at exactly zero - never moves on its own. Locking the view
                            // to it keeps the system centered on screen even while individual
                            // particles get flung outward by gravity (or the dominant body
                            // carries a small residual drift balanced by far-flung, harder to
                            // see ejecta). Only the particles/marker shift with the camera;
                            // the nebula backdrop is drawn unshifted, as if at infinity.
                            const com = computeCenterOfMass(this.particles);
                            const offsetX = sketch.width / 2 - com.x;
                            const offsetY = sketch.height / 2 - com.y;

                            if (this.debugPanelVisible) {
                                // Relative to the FIXED spawn-time world center (this.worldCenter),
                                // not the live canvas dimensions the camera uses - reusing the
                                // camera's offset here was the bug: resizing the browser window
                                // (or just toggling devtools) changes sketch.width/height, which
                                // silently redefines "center" for this measurement with no
                                // particle having actually moved, masquerading as huge drift.
                                this.centerOfMass.x = com.x - this.worldCenter.x;
                                this.centerOfMass.y = com.y - this.worldCenter.y;
                            }

                            sketch.push();
                            sketch.translate(offsetX, offsetY);

                            for (let particle of this.particles) {
                                particle.display(this.texturesEnabled);
                            }

                            for (let explosion of this.explosions) {
                                explosion.display(sketch);
                            }

                            // Marks the center of mass - should render pinned to the middle
                            // of the screen every frame, since the camera is centered on it.
                            if (this.crosshairVisible) {
                                sketch.stroke(255);
                                sketch.strokeWeight(1.5);
                                sketch.line(com.x - 8, com.y, com.x + 8, com.y);
                                sketch.line(com.x, com.y - 8, com.x, com.y + 8);
                            }

                            sketch.pop();
                        }

                    };

                });
            },
        },
    };
</script>

<style scoped>
    .simulation {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
    }

    .debug-panel {
        position: fixed;
        bottom: 20px;
        left: 20px;
        padding: 10px 14px;
        background: rgba(10, 8, 22, 0.55);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        color: #e8e2ff;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
    }

    .debug-title {
        color: #b9a6ff;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }

    .settings-panel {
        position: fixed;
        bottom: 72px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 12px 16px;
        background: rgba(10, 8, 22, 0.55);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        color: #e8e2ff;
        font-size: 13px;
        max-height: 70vh;
        overflow-y: auto;
    }

    .settings-row {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        white-space: nowrap;
    }

    .settings-row input[type='checkbox'],
    .settings-row input[type='radio'] {
        width: 14px;
        height: 14px;
        accent-color: #b06cff;
        cursor: pointer;
    }

    .settings-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding-top: 8px;
        border-top: 1px solid rgba(138, 92, 246, 0.2);
    }

    .settings-label {
        color: #b9a6ff;
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .controls {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        max-width: 94vw;
        gap: 10px;
        padding: 10px 14px;
        background: rgba(10, 8, 22, 0.45);
        border: 1px solid rgba(138, 92, 246, 0.25);
        border-radius: 10px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }

    .controls button {
        padding: 8px 16px;
        border: 1px solid rgba(138, 92, 246, 0.5);
        border-radius: 6px;
        background: linear-gradient(180deg, rgba(48, 32, 78, 0.85), rgba(18, 12, 32, 0.85));
        color: #e8e2ff;
        font-size: 13px;
        letter-spacing: 0.03em;
        cursor: pointer;
        box-shadow: 0 0 8px rgba(138, 92, 246, 0.25);
        transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.1s ease;
    }

    .controls button:hover {
        border-color: rgba(190, 130, 255, 0.9);
        box-shadow: 0 0 14px rgba(170, 110, 255, 0.5);
    }

    .controls button:active {
        transform: scale(0.96);
    }

    .controls button.active {
        border-color: rgba(255, 221, 51, 0.8);
        color: #fff6d8;
        box-shadow: 0 0 14px rgba(255, 221, 51, 0.45);
    }

    /* Mobile: fewer, smaller buttons (Central Mass moved into the settings panel) keep the
       control bar on one row instead of wrapping, and the debug readout moves out of the
       way of the bottom UI cluster entirely. */
    @media (max-width: 600px) {
        .controls {
            gap: 6px;
            padding: 6px 8px;
            bottom: 12px;
            flex-wrap: nowrap;
            max-width: 96vw;
        }

        .controls button {
            padding: 6px 10px;
            font-size: 11px;
            flex: 0 1 auto;
            white-space: nowrap;
        }

        .settings-panel {
            bottom: 62px;
            max-width: 90vw;
            padding: 10px 14px;
        }

        .debug-panel {
            top: 16px;
            left: 16px;
            bottom: auto;
            padding: 8px 12px;
            font-size: 11px;
        }
    }
</style>
