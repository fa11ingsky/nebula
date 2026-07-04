// Barnes-Hut quadtree: spatial index used by both gravity.ts (force approximation) and
// merge.ts (nearby-particle range queries).
import constants from '../constants.ts';
import { computeCenterOfMass } from './particleSystem.ts';

/**
 * A Barnes-Hut quadtree node: a square region of space that either holds a single
 * particle directly (or, past QUADTREE_MAX_DEPTH, a small flat bucket of coincident-ish
 * particles), or has been subdivided into four children. Every node - leaf or internal -
 * tracks the total mass and center of mass of everything inside it, accumulated
 * incrementally as particles are inserted, which is what lets a distant node be treated
 * as a single aggregate body during force/range queries instead of visiting its contents
 * one by one. Nodes store particle INDICES (into a ParticleSystem), not object
 * references - -1 means "empty".
 */
export class QuadTreeNode {
    constructor(x, y, size) {
        this.reset(x, y, size);
    }

    // Reinitializes this node in place so it can be pulled from a pool and reused
    // instead of allocated fresh - see acquireQuadTreeNode. At tens of thousands of
    // particles, a tree can easily involve 100K+ node objects; recreating all of them
    // from scratch every frame is a major source of GC pressure at that scale.
    reset(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.occupant = -1;
        this.bucket = null;
        this.children = null;
        this.mass = 0;
        this.comX = 0;
        this.comY = 0;
        return this;
    }

    insert(system, index, depth) {
        // Every node on the insertion path folds this particle into its running
        // mass-weighted center of mass, regardless of whether it ends up a leaf or
        // internal node - this is what makes "treat this whole subtree as one point
        // mass" possible later.
        const mass = system.mass[index];
        const px = system.posX[index];
        const py = system.posY[index];

        if (this.mass === 0) {
            this.comX = px;
            this.comY = py;
        } else {
            const newMass = this.mass + mass;
            this.comX = (this.comX * this.mass + px * mass) / newMass;
            this.comY = (this.comY * this.mass + py * mass) / newMass;
        }
        this.mass += mass;

        if (this.children) {
            this.childFor(system, index).insert(system, index, depth + 1);
            return;
        }

        if (this.bucket) {
            this.bucket.push(index);
            return;
        }

        if (this.occupant === -1) {
            this.occupant = index;
            return;
        }

        if (depth >= constants.QUADTREE_MAX_DEPTH) {
            // Particles landed on (almost) the same point and there's no room left to
            // subdivide meaningfully - fall back to a flat list instead of recursing forever.
            this.bucket = [this.occupant, index];
            this.occupant = -1;
            return;
        }

        this.subdivide();
        const existing = this.occupant;
        this.occupant = -1;
        this.childFor(system, existing).insert(system, existing, depth + 1);
        this.childFor(system, index).insert(system, index, depth + 1);
    }

    subdivide() {
        const half = this.size / 2;
        this.children = [
            acquireQuadTreeNode(this.x, this.y, half),              // NW
            acquireQuadTreeNode(this.x + half, this.y, half),        // NE
            acquireQuadTreeNode(this.x, this.y + half, half),        // SW
            acquireQuadTreeNode(this.x + half, this.y + half, half), // SE
        ];
    }

    // Uses the particle's clamped treeX/treeY (see buildQuadtree) rather than its true
    // position - that's what keeps a single far-flung outlier from forcing the whole
    // tree (and every normal, tightly-packed particle in it) through extra levels of
    // subdivision just to accommodate one point way out on its own.
    childFor(system, index) {
        const half = this.size / 2;
        const east = system.treeX[index] >= this.x + half ? 1 : 0;
        const south = system.treeY[index] >= this.y + half ? 1 : 0;
        return this.children[south * 2 + east];
    }
}

// Reused across every tree build instead of letting each one allocate 100K+ fresh node
// objects at high particle counts. Safe because tree builds within a frame are always
// sequential, never concurrent: buildQuadtree() resets the cursor to 0 at the start of
// each call, which only happens once the *previous* tree is done being read (merge
// detection finishes before gravity's tree is built or reused - see simulation.ts).
const quadTreeNodePool = [];
let quadTreeNodePoolCursor = 0;

function acquireQuadTreeNode(x, y, size) {
    let node = quadTreeNodePool[quadTreeNodePoolCursor];
    if (node) {
        node.reset(x, y, size);
    } else {
        node = new QuadTreeNode(x, y, size);
        quadTreeNodePool.push(node);
    }
    quadTreeNodePoolCursor++;
    return node;
}

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
export function buildQuadtree(system) {
    // Reclaim every node from the last tree built - safe per the pooling note above.
    quadTreeNodePoolCursor = 0;

    const com = computeCenterOfMass(system);

    let sumSqDeviation = 0;
    for (let i = 0; i < system.count; i++) {
        const dx = system.posX[i] - com.x;
        const dy = system.posY[i] - com.y;
        sumSqDeviation += dx * dx + dy * dy;
    }
    const rmsSpread = system.count > 0 ? Math.sqrt(sumSqDeviation / system.count) : 1;

    const halfSize = Math.max(rmsSpread * 6, 10);
    const size = halfSize * 2;
    const rootX = com.x - halfSize;
    const rootY = com.y - halfSize;
    const maxCoord = rootX + size * (1 - 1e-9); // just inside the far boundary

    for (let i = 0; i < system.count; i++) {
        system.treeX[i] = Math.min(Math.max(system.posX[i], rootX), maxCoord);
        system.treeY[i] = Math.min(Math.max(system.posY[i], rootY), maxCoord);
    }

    const root = acquireQuadTreeNode(rootX, rootY, size);
    for (let i = 0; i < system.count; i++) {
        root.insert(system, i, 0);
    }
    return root;
}
