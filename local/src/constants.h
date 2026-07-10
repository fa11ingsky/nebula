#pragma once
// Mirrors the subset of src/lib/constants.ts this native port actually uses - no merging,
// no textures/explosion flashes, no distributed-central-mass variant, so their constants
// (MIN_MERGE_VELOCITY, EXPLOSION_*, texture toggles) aren't carried over.
#include <cstdint>

namespace constants {
    constexpr int32_t MAX_PARTICLES = 100000; // native app has no WASM-heap capacity ceiling to mirror - just generously sized
    constexpr float MAX_MASS = 500.f;
    constexpr float GRAVITATIONAL_CONSTANT = 1.f;
    // PM gravity (pm_gravity.h, --pm) solves d^2(phi) = 4*pi*G*rho directly on this sim's 2D
    // grid, which gives a true 2D force law (~1/r, logarithmic potential) - not the tree
    // path's 3D-style ~1/r^2 law applied to these same 2D positions (see gravity.h). 1/r and
    // 1/r^2 aren't just different *shapes*, they're a different overall *scale* at this app's
    // actual mass/distance units, so this needs its own constant, tuned empirically rather
    // than derived - not a physically derived equivalence (no single constant can make two
    // different force laws match at every distance simultaneously).
    //
    // Re-tuned after fixing a real bug where computeGravityPMMesh never zeroed accX/accY
    // itself (see that function's own comment) - acceleration silently accumulated without
    // bound every frame, which is what the *previous* value of this constant (0.00002) was
    // unknowingly calibrated against. Once fixed, that value produced no visible clustering
    // at all even after a 100x increase (0.00002 -> 0.002) - self-gravitating collapse
    // against the swarm's own angular momentum is a threshold effect, not a smooth gradual
    // scaling, so small multiplicative bumps can look like "nothing's happening" right up
    // until a much larger one suddenly overshoots (confirmed directly: G=1 caused
    // catastrophic near-instant collapse). 0.02 was bisected to match the tree path's own
    // default clustering rate (avgCandidates/particle) over the same observation window.
    constexpr float PM_GRAVITATIONAL_CONSTANT = 0.02f;
    // P3M short-range correction (pm_gravity.h's PMShortRangeTable/applyP3MCorrection) - the
    // real fix for PM's confirmed close-range force error (see pm_gravity.h's header
    // comment), not just the Gaussian-softening mitigation. PM_P3M_CUTOFF_FACTOR sets how
    // far out (in units of sqrt(cell area)) the correction reaches before the mesh alone is
    // trusted again. P3M cost scales with this SQUARED (reach sets the neighbor-search area,
    // so pair count per particle grows quadratically with it) - the original 6.0, paired
    // with a soft (3-cell) mesh, measured 437ms/solve at 50k particles, nearly all P3M
    // pairs. Now paired with a sharper mesh (pm_gravity.h's PM_MESH_SOFTENING_CELLS = 1.5,
    // which becomes accurate at shorter range and so needs less P3M reach) - whatever error
    // the sharper mesh has *inside* this cutoff is exactly what the calibration table
    // cancels, so this pairing trades cost, not correctness. PM_PAIR_SOFTENING_FACTOR
    // is the direct-pairwise replacement force's own close-in softening (combinedRadius^2 *
    // this factor added under the sqrt) - same role as GRAVITY_SOFTENING_FACTOR for the tree
    // path, sized off particle radius so two just-touching particles don't see a divergent
    // 1/r as their separation approaches zero. PM_CALIBRATION_TABLE_SIZE is the resolution of
    // the one-time startup table that calibrates the correction against whatever the mesh
    // solve actually produces (see PMShortRangeTable::calibrate).
    constexpr float PM_P3M_CUTOFF_FACTOR = 4.f;
    constexpr float PM_PAIR_SOFTENING_FACTOR = 1.f;
    constexpr int32_t PM_CALIBRATION_TABLE_SIZE = 64;
    // Plummer softening for the tree path's direct (leaf) pairs, as a multiple of the
    // pair's combined radius squared - see constants.ts. The native port briefly carried
    // 200000 here (a 100x transcription slip against the web app's 2000, most visible as
    // "2 particles + --central-mass never gravitate": with N=2 each particle has mass 250
    // and radius ~15.8, so combinedRadius^2 * 200000 gave a ~14,000px softening length -
    // larger than the whole world, flattening the central mass's pull to ~zero at any
    // onscreen distance. Verified headless: the pair flew off in straight lines at spawn
    // speed with energy exactly conserved - no force, not an integrator leak.)
    constexpr float GRAVITY_SOFTENING_FACTOR = 2000.f;
    // Cap on the softening LENGTH (pixels) that GRAVITY_SOFTENING_FACTOR can produce.
    // Radius-proportional softening assumes swarm-sized particles (radius ~0.45 at the
    // default 2500 count -> ~40px softening length, same as the web app); it breaks down
    // when per-particle mass is huge (radius = sqrt(MAX_MASS/count), so a 2-particle run
    // gets 15.8px radii and a 31.6*sqrt(2000) = 1414px softening length - ~3.5x the typical
    // orbit radius, measured to leave the pair unbound even with the correct factor).
    // Capping the length at 200px leaves every normal swarm pair untouched (at the default
    // 2500 count even the central mass spawns with the swarm's own 0.45px radius - see
    // spawn.h's radiusOverride - so all softening lengths sit at ~40px, far under the cap;
    // verified bit-identical dynamics with and without it) while letting heavy bodies pull
    // with real ~1/r^2 strength beyond 200px. 200 rather than 100 because the N=2 orbit's
    // circular-velocity match is best there (measured: apoapsis ~800px vs ~2400px at 100).
    constexpr float GRAVITY_SOFTENING_MAX_LENGTH = 200.f;
    // Web app's value (constants.ts derives ~95,200 as the Keplerian balance point for its
    // canvas; this window's geometry works out to ~87k, same ballpark). The port briefly
    // carried 200000 - identical to the softening slip above, adjacent lines - which spawns
    // particles at ~2x the balance speed. That's ABOVE escape velocity for the few-particle
    // + central-mass configuration (v0 ~1.6 vs v_esc ~1.3 even with zero softening), so 2
    // orbiting particles drifted away no matter how gravity was softened - both this and
    // the softening had to be fixed together (verified: either one alone still escapes).
    constexpr float TOTAL_ANGULAR_MOMENTUM = 950.f;
    constexpr float BARNES_HUT_THETA = 0.2f;
    constexpr int32_t QUADTREE_MAX_DEPTH = 16;
    constexpr int32_t QUADTREE_LEAF_CAPACITY = 16;
    constexpr int32_t COLLISION_TREE_MAX_DEPTH = 18;
    constexpr int32_t COLLISION_TREE_LEAF_CAPACITY = 4;
    constexpr float SPAWN_RADIUS_FRACTION = 0.95f;
    constexpr float CENTRAL_MASS_FRACTION = 0.2f;
    constexpr float COLLISION_RESTITUTION = 0.5f;
    constexpr float COLLISION_SURFACE_GAP = 0.f;
    constexpr int32_t GRAVITY_MAX_THREADS = 16;
    constexpr int32_t COLLISION_MAX_THREADS = 16;
    // Rendering only (renderer.h): a particle's collision candidate count (collide.h's
    // CollisionCandidates.counts - already computed every frame, free to reuse as a local-
    // density proxy) divided by this and clamped to 1 gives the 0..1 "how blurred should
    // this particle look" factor. Lower = blur kicks in at lighter crowding; this was picked
    // to land in the same ballpark as the avgCandidates/particle range this port's own
    // profiling already treats as "visibly dense" (see main.cpp's debug printf).
    constexpr float DENSITY_BLUR_THRESHOLD = 2.f;
    // Adaptive substepping (main.cpp): a fixed dt=1 kick can overshoot badly when a particle
    // sees strong, fast-varying acceleration (a close pass near the central mass or a dense
    // clump) - the discrete kick projects the start-of-step acceleration in a straight line
    // for the whole frame, missing the true curved trajectory, injecting spurious kinetic
    // energy each such pass ("numerical slingshot" - see the explanation this was built to
    // address). The fix isn't finer position integration alone (that keeps using the same
    // stale acceleration) - it's re-deriving acceleration more often, i.e. running the whole
    // kick-drift-collide-gravity cycle more times per rendered frame, each with a smaller dt,
    // specifically when the peak acceleration this frame warrants it.
    //
    // SUBSTEP_SAFETY_FACTOR bounds how far pure acceleration (no initial velocity) could
    // move a particle within one substep, as a fraction of that particle's own radius -
    // reusing "combinedRadius"-scale reasoning the same way collide.ts/resolveOverlap sizes
    // its own touch/merge thresholds off particle radius rather than a fixed pixel value.
    // Smaller = more conservative (more substeps sooner); MAX_SUBSTEPS caps the worst-case
    // per-frame cost blowup from a truly extreme close encounter.
    constexpr float SUBSTEP_SAFETY_FACTOR = 0.5f;
    constexpr int32_t MAX_SUBSTEPS = 8;
}
