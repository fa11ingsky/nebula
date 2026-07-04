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
    TOTAL_PARTICLES: 2500,
    GRAVITY: {
        X: 0,
        Y: 0
    },
    // With Central Mass enabled, this body dominates the gravity field (it outweighs an
    // average swarm particle by ~500x), so the swarm effectively orbits it like a sun.
    // 1 is the balance point: strong enough to bind and accrete the swarm at this canvas's
    // mass/distance scale, without collapsing so violently that everything merges before a
    // "system" is recognizable.
    GRAVITATIONAL_CONSTANT: 1,
    // Plummer softening, expressed as a multiple of the interacting pair's own combined
    // radius rather than a fixed pixel length. That way, as particles grow through merging,
    // gravity stays proportionate to their actual size instead of treating an ever-larger
    // merged body as an ever-more-extreme point mass.
    // Solved from the merge geometry, not guessed: two particles merge once separated by
    // less than minDistance = combinedRadius/2 (see mergeParticles). Softening should only
    // matter as a numerical safety net *inside* that distance (protecting against a rare
    // single-frame tunneling pass before the merge check fires) and stay out of the way of
    // real, unmerged particles interacting normally. Setting the softening length equal to
    // the merge threshold itself - i.e. combinedRadius^2 * FACTOR = (combinedRadius/2)^2 -
    // gives FACTOR = 1/4. The old value of 1 suppressed the force to ~9% of its true
    // Newtonian strength right at the merge boundary (softenedDistSq = 1.25*combinedRadius^2
    // there, versus the "true" 0.25*combinedRadius^2); at 0.25, force at that same boundary
    // is ~35% of unsoftened strength and rises to the full, correct value for any separation
    // beyond it - so softening only really bites in the narrow zone below the merge distance
    // that shouldn't normally be reached at all.
    GRAVITY_SOFTENING_FACTOR: 200,
    // Net angular momentum (about the system's center of mass) that initial particle velocities are scaled to produce.
    // Sign controls spin direction; magnitude controls how fast the system orbits before gravity reshapes it.
    // Solved, not guessed: the initial velocity field is a rigid-body rotation (v = omega * r),
    // not a true Keplerian profile (v ~ 1/sqrt(r)), so setting omega*r_rms equal to the
    // Keplerian circular velocity around the central mass at the swarm's characteristic
    // (RMS) radius r_rms gives L = MAX_MASS * sqrt(GRAVITATIONAL_CONSTANT * centralMass) *
    // 2^-0.25 * sqrt(spawnRadius). With this canvas's mass scale (central mass =
    // MAX_MASS * CENTRAL_MASS_FRACTION = 100) and a representative 1920x1080 viewport
    // (spawnRadius ~= 513), that works out to ~95,200. This isn't "perfectly circular for
    // everyone" - it can't be, given the v=omega*r field - it's the point where inner
    // particles trending slightly slow (spiraling in) and outer ones trending slightly fast
    // (eccentric) are balanced around zero, which is exactly the mix of crossing orbits that
    // lets particles collide and accrete into "planets" instead of either all plunging into
    // one blob (too little L) or all settling into wide, non-interacting orbits (too much L).
    TOTAL_ANGULAR_MOMENTUM: 95000,
    // Choices offered in the settings panel for each of these three - add/remove/reorder
    // entries here to change what shows up as radio options, no template changes needed.
    ANGULAR_MOMENTUM_OPTIONS: [10000, 50000, 95000, 150000],
    GRAVITATIONAL_CONSTANT_OPTIONS: [0.1, 0.5, 1, 2],
    TOTAL_PARTICLES_OPTIONS: [100, 1000, 2500, 4000, 8000, 10000, 25000, 50000],
    // Barnes-Hut opening angle: a distant cluster of particles is treated as one point mass
    // at its center of mass once (node size / distance) < this, instead of visiting every
    // particle inside it individually. 0.5 is the classic Barnes & Hut (1986) value - a good
    // default balance between speed and force accuracy. Lower = more accurate, slower;
    // higher = faster, coarser (fewer node visits per particle, since aggregation kicks in
    // sooner) - worth raising as particle count climbs into the tens of thousands.
    BARNES_HUT_THETA: 0.5,
    BARNES_HUT_THETA_OPTIONS: [0.3, 0.5, 0.8, 1.2, 3],
    // Safety cap on quadtree subdivision depth, in case many particles land on (almost) the
    // same point - without this, that would try to subdivide forever. Nodes deeper than this
    // just keep a flat list instead of recursing further.
    QUADTREE_MAX_DEPTH: 24,
    // Particles spawn within this fraction of half the canvas size, keeping the initial
    // cluster comfortably inside the visible area.
    SPAWN_RADIUS_FRACTION: 0.95,
    // Mass of the optional central body, as a fraction of MAX_MASS (the swarm's total mass).
    CENTRAL_MASS_FRACTION: 0.2,
    // Two particles within merge distance only actually fuse if their relative (closing)
    // speed is at least this - a slow graze just passes through without merging. This is a
    // deliberately small default: it's meant to filter out only the gentlest near-misses
    // (e.g. two bodies briefly touching while drifting past each other in similar orbits),
    // not to block normal accretion. Tune it up if too many low-speed touches are still
    // fusing, or down toward 0 to go back to "any contact merges."
    MIN_MERGE_VELOCITY: 1,
    // How many frames a collision flash lasts, and how large it grows relative to
    // sqrt(combined mass) - i.e. proportional to the merged body's own radius.
    EXPLOSION_DURATION_FRAMES: 240,
    EXPLOSION_RADIUS_FACTOR: 2
}
