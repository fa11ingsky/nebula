export default {
    MAX_MASS: 500,
    // Six-stop gradient a particle sweeps through as it gains mass, smallest to heaviest.
    COLORS: {
        BROWN: [139, 69, 19],
        BLUE: [59, 130, 246],
        GREEN: [50, 200, 90],
        RED: [220, 40, 40],
        YELLOW: [255, 221, 51],
        WHITE: [255, 255, 255]
    },
    TOTAL_PARTICLES: 2000,
    GRAVITY: {
        X: 0,
        Y: 0
    },
    GRAVITATIONAL_CONSTANT: 1,
    // Plummer softening, expressed as a multiple of the interacting pair's own combined
    // radius rather than a fixed pixel length. That way, as particles grow through merging,
    // gravity stays proportionate to their actual size instead of treating an ever-larger
    // merged body as an ever-more-extreme point mass.
    GRAVITY_SOFTENING_FACTOR: 4,
    // Net angular momentum (about the system's center of mass) that initial particle velocities are scaled to produce.
    // Sign controls spin direction; magnitude controls how fast the system orbits before gravity reshapes it.
    // Raised toward the system's rough virial value (where centrifugal motion balances
    // gravity) so particles settle into calmer, more circular orbits instead of head-on
    // plunges - those plunges are what was flinging particles out via close encounters.
    TOTAL_ANGULAR_MOMENTUM: 200000,
    // Particles spawn within this fraction of half the canvas size, keeping the initial
    // cluster comfortably inside the visible area.
    SPAWN_RADIUS_FRACTION: 0.95,
    // Mass of the optional central body, as a fraction of MAX_MASS (the swarm's total mass).
    CENTRAL_MASS_FRACTION: 0.4,
    // How many frames a collision flash lasts, and how large it grows relative to
    // sqrt(combined mass) - i.e. proportional to the merged body's own radius.
    EXPLOSION_DURATION_FRAMES: 120,
    EXPLOSION_RADIUS_FACTOR: 5
}
