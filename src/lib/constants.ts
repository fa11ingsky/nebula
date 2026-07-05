export default {
    // Shown in the debug panel - bump manually as a quick way to tell, at a glance,
    // whether a deployed build actually picked up recent changes.
    VERSION: '0.1.2',
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
    GRAVITY_SOFTENING_FACTOR: 2000,
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
    QUADTREE_MAX_DEPTH: 10,
    // WASM-only (see src/wasm/gravity.cpp): stop subdividing once a node's particle count
    // drops to this many or fewer, treating them as one direct-summation leaf instead of
    // continuing to split down to single-occupant nodes. Subdivision itself has overhead
    // (partitioning, allocating child nodes, an extra traversal level for every visitor),
    // so for a small enough batch, a handful of direct pairwise force calculations is
    // cheaper than isolating each particle into its own leaf. Too high wastes time on
    // direct pairwise checks a coarser aggregate could've approximated instead; too low
    // (down to 1) recovers the old single-occupant-leaf behavior at the cost of more
    // subdivisions. 8 is a common starting point in other Barnes-Hut implementations -
    // tune based on measured frame time at your actual particle counts.
    QUADTREE_LEAF_CAPACITY: 16,
    // Particles spawn within this fraction of half the canvas size, keeping the initial
    // cluster comfortably inside the visible area.
    SPAWN_RADIUS_FRACTION: 0.95,
    // The optional central mass isn't one dominant body - it's a dense cluster of
    // ordinary swarm particles (same per-particle mass as everyone else) spawned within a
    // small radius at the center instead of spread across the whole disc. This is that
    // cluster's share of the total particle count (and so, since every particle has equal
    // mass, of MAX_MASS too) - e.g. 0.2 with TOTAL_PARTICLES=2500 means 500 of those 2500
    // particles spawn packed into the center instead of scattered through the disc.
    CENTRAL_MASS_FRACTION: 0.2,
    // How tightly packed that central cluster spawns, as a fraction of the swarm's own
    // spawn radius - small enough to read as a dense "core" rather than just a denser
    // patch of the same disc, but not a literal single point (letting the cluster's own
    // particles start at slightly different positions avoids spawning many bodies exactly
    // coincident, which is otherwise harmless - softening and the quadtree's bucket
    // fallback both handle it - but looks like a visual glitch before gravity pulls them
    // into a natural clump anyway).
    CENTRAL_CLUSTER_RADIUS_FRACTION: 0.05,
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
    EXPLOSION_RADIUS_FACTOR: 2,
    // With merging disabled, colliding particles bounce off each other instead (see
    // collide.ts) - this is the coefficient of restitution for that bounce: the fraction
    // of closing speed preserved (reversed) afterward. 1 would be a perfectly elastic
    // billiard-ball collision (kinetic energy exactly conserved); 0.5 keeps only half the
    // closing speed, so each collision bleeds some kinetic energy as "heat" - closer to
    // how real solid material collides, and lets a jostling cluster gradually settle down
    // instead of bouncing at full energy indefinitely.
    COLLISION_RESTITUTION: 0.1,
    // Extra gap (in canvas pixels, on top of the two bodies' own radii) that collision
    // treats as "touching" and enforces as a hard minimum separation - see collide.ts's
    // resolveOverlap. Purely cosmetic: without it, two bounced circles can render with
    // their edges exactly coincident (or overlapping by a sub-pixel floating-point
    // residue), which reads as a visual glitch even though it's numerically correct.
    COLLISION_SURFACE_GAP: 2
}
