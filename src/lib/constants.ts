export default {
    // Shown in the debug panel - bump manually as a quick way to tell, at a glance,
    // whether a deployed build actually picked up recent changes.
    VERSION: '0.2.0',
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
    // Which gravity solver runs each frame - see the settings panel's "Gravity Solver"
    // group and simulation.worker.ts:
    //   'tree' - Barnes-Hut quadtree (gravity.ts), the original path. ~1/r^2 force law.
    //   'pm'   - Particle-Mesh on the CPU (pmGravity.ts, ported from the native app's
    //            local/src/pm_gravity.h): FFT Poisson solve + P3M short-range correction.
    //            A true-2D gravity model (~1/r force law) - different physics, not just a
    //            faster path; see pmGravity.ts's header comment.
    //   'gpu'  - the same PM+P3M model plus collision and integration, all as WebGPU
    //            compute shaders (webgpuSim.ts, ported from local/src/gpu_sim.h).
    //            Positions never leave GPU memory; this is the only solver that scales to
    //            the 250k/1M particle-count options. Requires WebGPU; forces the WebGPU
    //            render backend (physics and rendering share one GPU device/buffer).
    GRAVITY_SOLVER: 'tree',
    GRAVITY_SOLVER_OPTIONS: ['tree', 'pm', 'gpu'],
    // PM gravity's own gravitational constant - the PM mesh solves a true 2D Poisson
    // equation (~1/r force law, logarithmic potential), a different overall force scale at
    // this app's mass/distance units than the tree's ~1/r^2 law, so it can't share
    // GRAVITATIONAL_CONSTANT. This value was bisected in the native app to match the tree
    // path's clustering rate at the default swarm. The GRAVITATIONAL_CONSTANT setting
    // still applies as a multiplier on top (applied at spawn/init - PM modes bake G into
    // their Green's table and calibration table, so mid-run changes take effect on
    // Restart, unlike the tree path's immediate pickup).
    PM_GRAVITATIONAL_CONSTANT: 0.02,
    // How far out (in units of sqrt(cell area)) the P3M short-range correction reaches
    // before the mesh alone is trusted. Cost scales with this SQUARED. Mirrors the native
    // app's constants.h.
    PM_P3M_CUTOFF_FACTOR: 4,
    // The P3M direct-pair force's own close-in softening (combinedRadius^2 * this, added
    // under the sqrt) - same role as GRAVITY_SOFTENING_FACTOR for the tree path.
    PM_PAIR_SOFTENING_FACTOR: 1,
    // Resolution of the one-time startup table that calibrates the P3M correction against
    // whatever the mesh solve actually produces (see pmGravity.ts's calibrate).
    PM_CALIBRATION_TABLE_SIZE: 64,
    // Adaptive substepping for the 'gpu' solver (ported from the native app's main loop):
    // per-frame substep count is derived from last frame's peak acceleration so close
    // encounters get integrated finely without paying for it on calm frames. The safety
    // factor bounds how far pure acceleration may move a particle in one substep, as a
    // fraction of particle radius; MAX_SUBSTEPS caps the worst-case per-frame cost.
    SUBSTEP_SAFETY_FACTOR: 0.5,
    MAX_SUBSTEPS: 8,
    // Choices offered in the settings panel for each of these three - add/remove/reorder
    // entries here to change what shows up as radio options, no template changes needed.
    ANGULAR_MOMENTUM_OPTIONS: [10000, 50000, 95000, 150000],
    GRAVITATIONAL_CONSTANT_OPTIONS: [0.1, 0.5, 1, 2],
    // The six-figure options are realistically only usable with the 'gpu' gravity solver
    // (and its WebGPU rendering) - the tree/Canvas2D combination will crawl there.
    TOTAL_PARTICLES_OPTIONS: [100, 1000, 2500, 4000, 8000, 10000, 25000, 50000, 100000, 250000, 1000000],
    // Barnes-Hut opening angle: a distant cluster of particles is treated as one point mass
    // at its center of mass once (node size / distance) < this, instead of visiting every
    // particle inside it individually. 0.5 is the classic Barnes & Hut (1986) value - a good
    // default balance between speed and force accuracy. Lower = more accurate, slower;
    // higher = faster, coarser (fewer node visits per particle, since aggregation kicks in
    // sooner) - worth raising as particle count climbs into the tens of thousands.
    BARNES_HUT_THETA: 0.5,
    BARNES_HUT_THETA_OPTIONS: [0.3, 0.5, 0.8, 1.2, 3],
    // WASM-only (see src/wasm/gravity.cpp's compute_gravity): ceiling on how many
    // std::thread workers the threaded WASM build spawns for gravity's per-particle force
    // loop, on TOP of the browser's own navigator.hardwareConcurrency (this never spawns
    // more threads than the browser reports logical cores for, regardless of how high this
    // is set). Raising this past your actual core count won't add throughput - there's no
    // more CPU to schedule extra threads onto - it only adds cost, since every thread here
    // is spawned fresh and joined again up to 60 times a second; oversubscribing just means
    // paying that spawn/join overhead for threads that end up time-slicing against each
    // other on the same cores. Has no effect at all on the non-threaded WASM build (always
    // runs sequentially there) or when the page isn't cross-origin isolated.
    GRAVITY_MAX_THREADS: 16,
    // Same idea as GRAVITY_MAX_THREADS, but for collide.ts's broad-phase search (see
    // gravity.cpp's find_all_collision_candidates) - kept separate since profiling showed
    // this search, not gravity's own traversal, is actually the majority of a collision
    // frame's cost, so it's the one most worth tuning independently if you're
    // experimenting with thread counts on your own hardware.
    COLLISION_MAX_THREADS: 16,
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
    // WASM-only (see src/wasm/gravity.cpp's build_collision_tree): the same idea as
    // QUADTREE_LEAF_CAPACITY, but for the separate tree collide.ts's broad-phase search
    // builds - kept as its own constant because the right tuning point is different for
    // the two purposes. Gravity wants coarser leaves (a bigger direct-summation batch is
    // fine, since every member contributes real force regardless of whether it's nearby or
    // just node-adjacent). Collision wants finer leaves: every candidate a leaf returns
    // still has to run through collide.ts's exact swept-collision math, so an oversized
    // leaf mostly just hands back particles that were never actually going to touch -
    // wasted work in JS, not saved work in WASM. Small (2-8) is the right range here.
    COLLISION_TREE_LEAF_CAPACITY: 4,
    // Separate max-depth safety cap for that same collision tree, independent of
    // QUADTREE_MAX_DEPTH - measured directly, sharing one depth cap between the two trees
    // was actively wrong: QUADTREE_MAX_DEPTH is tuned low (10) for gravity's tree, where a
    // shallow cap is fine since aggregation handles distant nodes regardless of depth. The
    // collision tree has no such escape hatch - COLLISION_TREE_LEAF_CAPACITY only actually
    // bounds leaf size if there's enough depth budget to subdivide that far down. Profiling
    // with the distributed central-mass feature enabled (which packs particles into a much
    // denser cluster than gravity's tree ever has to fully separate) showed leaves still
    // holding 72 particles at depth 10, versus correctly settling at ~4 by depth 18 - so
    // this needs to go deep enough for the density collision actually has to handle, not
    // whatever's fast enough for gravity's very different access pattern.
    COLLISION_TREE_MAX_DEPTH: 18,
    // Particles spawn within this fraction of half the canvas size, keeping the initial
    // cluster comfortably inside the visible area.
    SPAWN_RADIUS_FRACTION: 0.95,
    // The optional central mass is a single fixed particle (see particleSystem.ts's `fixed`
    // field and spawn.ts) planted exactly at the arena center, holding this fraction of
    // MAX_MASS - e.g. 0.2 with MAX_MASS=500 gives it a mass of 100, same total mass the
    // swarm's own particles sum to. It attracts and can be collided with like any other
    // body, it just never moves itself, acting as a fixed anchor the swarm orbits/accretes
    // around rather than one more free body in the system.
    CENTRAL_MASS_FRACTION: 0.1,
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
    COLLISION_SURFACE_GAP: 0
}
