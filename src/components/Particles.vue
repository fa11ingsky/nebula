<template>
    <div ref="canvasContainer"></div>
    <div>
        <button id="stopButton" @click="stopped = !stopped">Stop</button>
        <button ref="restartButton" @click="resetSim()">Restart</button>
    </div>
</template>

<script>
    import p5 from 'p5';
    import constants from '../lib/constants.ts';

    class Particle {
        constructor(x, y, s) {
            this.position = s.createVector(x, y);
            this.velocity = s.createVector(-1, -1);
            //this.velocity = sketch.createVector(sketch.random(-2, 2), sketch.random(-2, 2)); // Random initial velocity
            this.mass = constants.MAX_MASS / constants.TOTAL_PARTICLES;
            this.radius = Math.sqrt(this.mass); // Particle radius
            this.color = this.getColor(this.mass);
            this.acceleration = s.createVector(constants.GRAVITY.X, constants.GRAVITY.Y);
            this.s = s;
        }

        update() {
            this.position.add(this.velocity);

            // Bounce off edges
            let edge_pad = 6;
            if (this.position.x - this.radius < 0 + edge_pad || this.position.x + this.radius > this.s.width - edge_pad) {
                this.velocity.x *= -1;
            }
            if (this.position.y - this.radius< 0 + edge_pad || this.position.y  + this.radius > this.s.height - edge_pad) {
                this.velocity.y *= -1;
            }

            // Update velocity based on force
            this.velocity.add(this.acceleration);

        }

        merge(other) {
            // Combine particles based on momentum conservation
            let totalMass = this.mass + other.mass;
            let combinedVelocity = p5.Vector.add(p5.Vector.mult(this.velocity, this.mass), p5.Vector.mult(other.velocity, other.mass)).div(totalMass);
            this.position = this.position.copy().add(this.velocity.copy().mult(this.radius)).lerp(other.position.copy().add(other.velocity.copy().mult(other.radius)), this.mass / totalMass);
            this.velocity = combinedVelocity;
            this.radius = Math.sqrt(totalMass); // Adjust radius based on combined mass
            console.log(`Merge: ${this.radius}`)
            this.mass = totalMass
            this.color = this.getColor(this.mass)
        }

        display() {
            this.s.fill(this.color);
            this.s.noStroke();
            this.s.ellipse(this.position.x, this.position.y, this.radius * 2);
        }

        getColor(mass) {

            // Calculate the ratios between brown, blue, and yellow based on mass
            let brownRatio, blueRatio, yellowRatio;
            if (mass < 0.66 * constants.MAX_MASS) {
                brownRatio = 1;
                blueRatio = 0;
                yellowRatio = 0;
            } else if (mass < constants.MAX_MASS) {
                brownRatio = (constants.MAX_MASS - mass) / constants.MAX_MASS;
                blueRatio = 1;
                yellowRatio = 0;
            } else {
                brownRatio = 0;
                blueRatio = (mass - constants.MAX_MASS) / 100;
                yellowRatio = 1;
            }

            // Interpolate color based on ratios
            const r = Math.round(constants.COLORS.BROWN[0] * brownRatio + constants.COLORS.BLUE[0] * blueRatio + constants.COLORS.YELLOW[0] * yellowRatio);
            const g = Math.round(constants.COLORS.BROWN[1] * brownRatio + constants.COLORS.BLUE[1] * blueRatio + constants.COLORS.YELLOW[1] * yellowRatio);
            const b = Math.round(constants.COLORS.BROWN[2] * brownRatio + constants.COLORS.BLUE[2] * blueRatio + constants.COLORS.YELLOW[2] * yellowRatio);

            // Return the interpolated color
            return `rgb(${r}, ${g}, ${b})`;
        }
    }// end Particle class

    export default {
        data() {
            return {
                stopped: false,
                canvas: null,
                particles: []
            };
        },

        mounted() {
            this.createCanvas();
        },
        methods: {
            resetSim() {
                this.canvas.clear();
                this.particles = [];
                // Create three particles
                for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
                    //particles.push(new Particle(sketch.random(sketch.width), sketch.random(sketch.height)));
                    this.particles.push(new Particle(this.canvas.random(this.canvas.width), this.canvas.random(this.canvas.height), this.canvas));
                }
            },
            createCanvas() {
                this.canvas = new p5((sketch) => {

                    sketch.setup = () => {
                        sketch.createCanvas(constants.SPACE.WIDTH, constants.SPACE.HEIGHT).parent(this.$refs.canvasContainer);

                        // Create three particles
                        for (let i = 0; i < constants.TOTAL_PARTICLES; i++) {
                            this.particles.push(new Particle(sketch.random(sketch.width), sketch.random(sketch.height), sketch));
                            //this.particles.push(new Particle(i + 90, i * 100));
                        }
                    };

                    sketch.draw = () => {
                        if (!this.stopped) {
                            sketch.background(255);

                            // Update and display particles
                            for (let particle of this.particles) {
                                particle.update();
                                particle.display();
                            }

                            // Merge particles
                            this.particles = mergeParticles(this.particles);
                        }

                    };

                    /**
                     * Check for merged particles and returns a new array of remaining particles
                     * @param {Array} particles - array of initial particles
                     * @returns {Array} - array of remaining merged particles
                     */
                    function mergeParticles(particles) {
                        let merged = [];

                        for (let i = particles.length - 1; i >= 0; i--) {
                            let currentParticle = particles[i];
                            let mergedParticle = null;

                            // Check for collisions and merge
                            for (let j = i - 1; j >= 0; j--) {
                                let otherParticle = particles[j];
                                let distanceSq = p5.Vector.sub(currentParticle.position, otherParticle.position).mag();
                                let minDistanceSq = currentParticle.radius / 2 + otherParticle.radius / 2;

                                if (distanceSq <= minDistanceSq) {
                                    // Merge particles
                                    otherParticle.merge(currentParticle);
                                    mergedParticle = currentParticle;
                                    break;
                                }
                            }

                            // Add current particle if not merged
                            if (!mergedParticle) {
                                //console.log(`Adding merged particle radius: ${currentParticle.radius}`)
                                merged.push(currentParticle);
                            }
                        }

                        return merged;
                    } // end of mergeParticles



                });
            },
        },
    };
</script>

<style scoped>
    /* Add your styles here if needed */
</style>