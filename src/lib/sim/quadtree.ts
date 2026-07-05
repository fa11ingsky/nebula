// Barnes-Hut quadtree: spatial index used by both gravity.ts (force approximation) and
// merge.ts (nearby-particle range queries via findNearbyParticles). collide.ts uses its
// own uniform spatial grid (spatialGrid.ts) instead - see that file for why.
//
// Nodes live in flat, parallel typed arrays (a struct-of-arrays "store") rather than as
// individually-allocated, pooled JS objects each holding an array of 4 child references.
// Two things fall out of that:
//
//  - Contiguous, cache-friendly storage: visiting node N and its children means touching
//    nearby indices in a handful of typed arrays, not chasing pointers through separate
//    heap objects scattered wherever the GC happened to place them.
//  - Every node gets a `next` pointer, computed once at creation time: "the next node to
//    visit if this whole subtree gets skipped" - either a sibling, or (for the last of 4
//    children) whatever the parent's own `next` was. That turns tree traversal (gravity,
//    energy, range queries) into a single flat loop with no recursion and no explicit
//    stack: skip a subtree by jumping straight to `next`, or descend into `children`.
//    (This is the same "escape index" trick production BVH/ray-tracing traversal uses.)
//
// Center-of-mass is NOT updated incrementally during insertion the way the previous
// object-based tree did (touching every ancestor on every particle's insertion path,
// O(n log n) total updates across a frame). Instead, `propagate()` runs once after all
// insertions, walking already-built internal nodes bottom-up and computing each one's
// aggregate directly from its 4 children - O(1) work per internal node, visited exactly
// once, so O(n) total instead of O(n log n).
import constants from '../constants.ts';
import { computeCenterOfMass } from './particleSystem.ts';

const INITIAL_NODE_CAPACITY = 1024;

function growTypedArray(oldArray, ArrayType, newCapacity) {
    const newArray = new ArrayType(newCapacity);
    newArray.set(oldArray);
    return newArray;
}

function createStore(capacity) {
    return {
        capacity,
        count: 0,
        // Node geometry: (x, y) is the top-left corner, size is the full width/height.
        x: new Float32Array(capacity),
        y: new Float32Array(capacity),
        size: new Float32Array(capacity),
        // Index of the first of 4 contiguous children, or -1 if this node hasn't been
        // subdivided (it may still hold a single occupant or a depth-capped bucket).
        children: new Int32Array(capacity),
        // The traversal "escape" pointer described above; -1 means "nothing more to visit".
        next: new Int32Array(capacity),
        // Particle index held directly by this (unsubdivided) node, or -1.
        occupant: new Int32Array(capacity),
        // Overflow list for particles that landed on (almost) the same point past
        // QUADTREE_MAX_DEPTH - rare enough that it's not worth flattening into typed arrays.
        bucket: new Array(capacity).fill(null),
        // Aggregate mass / center of mass - valid for every node once buildQuadtree
        // finishes. Leaves get theirs set directly as particles are inserted; branch
        // nodes get theirs from propagate()'s bottom-up pass.
        mass: new Float32Array(capacity),
        comX: new Float32Array(capacity),
        comY: new Float32Array(capacity),
        // Every node that was subdivided this build, in the order subdivision happened -
        // propagate() walks this in reverse. A child can only be subdivided after its own
        // parent already was (subdivision is what creates the child in the first place),
        // so reversing this list always visits children before parents.
        parents: new Int32Array(capacity),
        parentCount: 0,
    };
}

function ensureCapacity(store, neededCount) {
    if (neededCount <= store.capacity) {
        return;
    }
    let newCapacity = store.capacity;
    while (newCapacity < neededCount) {
        newCapacity *= 2;
    }

    store.x = growTypedArray(store.x, Float32Array, newCapacity);
    store.y = growTypedArray(store.y, Float32Array, newCapacity);
    store.size = growTypedArray(store.size, Float32Array, newCapacity);
    store.children = growTypedArray(store.children, Int32Array, newCapacity);
    store.next = growTypedArray(store.next, Int32Array, newCapacity);
    store.occupant = growTypedArray(store.occupant, Int32Array, newCapacity);
    store.mass = growTypedArray(store.mass, Float32Array, newCapacity);
    store.comX = growTypedArray(store.comX, Float32Array, newCapacity);
    store.comY = growTypedArray(store.comY, Float32Array, newCapacity);
    store.parents = growTypedArray(store.parents, Int32Array, newCapacity);

    const newBucket = new Array(newCapacity).fill(null);
    for (let i = 0; i < store.bucket.length; i++) {
        newBucket[i] = store.bucket[i];
    }
    store.bucket = newBucket;

    store.capacity = newCapacity;
}

// Reused across every tree build instead of allocating fresh typed arrays 60 times a
// second - the same reuse-and-reset trick the old node pool used, just applied to a flat
// store instead of a pool of objects. `count`/`parentCount` reset to 0 at the start of
// each buildQuadtree call; the underlying arrays only ever grow, never shrink.
let store = createStore(INITIAL_NODE_CAPACITY);

function initNode(idx, x, y, size, next) {
    store.x[idx] = x;
    store.y[idx] = y;
    store.size[idx] = size;
    store.children[idx] = -1;
    store.next[idx] = next;
    store.occupant[idx] = -1;
    store.bucket[idx] = null;
    store.mass[idx] = 0;
    store.comX[idx] = 0;
    store.comY[idx] = 0;
}

function setLeafAggregate(node, system, particleIndex) {
    store.mass[node] = system.mass[particleIndex];
    store.comX[node] = system.posX[particleIndex];
    store.comY[node] = system.posY[particleIndex];
}

function addToLeafAggregate(node, system, particleIndex) {
    const m1 = store.mass[node];
    const m2 = system.mass[particleIndex];
    const totalMass = m1 + m2;
    store.comX[node] = (store.comX[node] * m1 + system.posX[particleIndex] * m2) / totalMass;
    store.comY[node] = (store.comY[node] * m1 + system.posY[particleIndex] * m2) / totalMass;
    store.mass[node] = totalMass;
}

// Uses the particle's clamped treeX/treeY (see buildQuadtree) rather than its true
// position - that's what keeps a single far-flung outlier from forcing the whole tree
// (and every normal, tightly-packed particle in it) through extra levels of subdivision
// just to accommodate one point way out on its own.
function findQuadrant(system, index, nodeX, nodeY, half) {
    const east = system.treeX[index] >= nodeX + half ? 1 : 0;
    const south = system.treeY[index] >= nodeY + half ? 1 : 0;
    return south * 2 + east;
}

function subdivide(node) {
    ensureCapacity(store, store.count + 4);
    const children = store.count;
    store.children[node] = children;
    store.parents[store.parentCount++] = node;

    const half = store.size[node] / 2;
    const x = store.x[node];
    const y = store.y[node];
    const parentNext = store.next[node];

    // NW, NE, SW, SE - matching findQuadrant's south*2+east ordering. Each child's `next`
    // is its right sibling, except the last (SE), whose `next` is the parent's own `next`
    // - i.e. "after this whole subtree, go wherever the parent would have gone".
    initNode(children + 0, x, y, half, children + 1);
    initNode(children + 1, x + half, y, half, children + 2);
    initNode(children + 2, x, y + half, half, children + 3);
    initNode(children + 3, x + half, y + half, half, parentNext);

    store.count += 4;
    return children;
}

function insert(system, index) {
    let node = 0;
    let depth = 0;

    while (store.children[node] !== -1) {
        const half = store.size[node] / 2;
        const quadrant = findQuadrant(system, index, store.x[node], store.y[node], half);
        node = store.children[node] + quadrant;
        depth++;
    }

    if (store.bucket[node] !== null) {
        store.bucket[node].push(index);
        addToLeafAggregate(node, system, index);
        return;
    }

    if (store.occupant[node] === -1) {
        store.occupant[node] = index;
        setLeafAggregate(node, system, index);
        return;
    }

    if (depth >= constants.QUADTREE_MAX_DEPTH) {
        // Particles landed on (almost) the same point and there's no room left to
        // subdivide meaningfully - fall back to a flat list instead of recursing forever.
        const existing = store.occupant[node];
        store.occupant[node] = -1;
        store.bucket[node] = [existing, index];
        addToLeafAggregate(node, system, index); // existing's contribution is already in place; fold the new one in
        return;
    }

    let existing = store.occupant[node];
    store.occupant[node] = -1;

    // Keep subdividing until the two particles land in different quadrants - equivalent
    // to the cascade of recursive insert() calls a tree-of-objects version would make,
    // just flattened into a loop instead of nested function calls.
    while (true) {
        const children = subdivide(node);
        const half = store.size[node] / 2;
        const q1 = findQuadrant(system, existing, store.x[node], store.y[node], half);
        const q2 = findQuadrant(system, index, store.x[node], store.y[node], half);
        depth++;

        if (q1 === q2) {
            node = children + q1;
            if (depth >= constants.QUADTREE_MAX_DEPTH) {
                store.bucket[node] = [existing, index];
                setLeafAggregate(node, system, existing);
                addToLeafAggregate(node, system, index);
                return;
            }
            continue;
        }

        store.occupant[children + q1] = existing;
        setLeafAggregate(children + q1, system, existing);
        store.occupant[children + q2] = index;
        setLeafAggregate(children + q2, system, index);
        return;
    }
}

/**
 * Computes every branch node's aggregate mass/center-of-mass from its 4 children, bottom-
 * up. Leaves already have theirs set directly during insertion (see setLeafAggregate/
 * addToLeafAggregate above); this only ever touches nodes that were subdivided.
 */
function propagate() {
    for (let p = store.parentCount - 1; p >= 0; p--) {
        const node = store.parents[p];
        const c = store.children[node];

        const m0 = store.mass[c];
        const m1 = store.mass[c + 1];
        const m2 = store.mass[c + 2];
        const m3 = store.mass[c + 3];
        const totalMass = m0 + m1 + m2 + m3;

        store.mass[node] = totalMass;
        if (totalMass > 0) {
            store.comX[node] = (store.comX[c] * m0 + store.comX[c + 1] * m1 + store.comX[c + 2] * m2 + store.comX[c + 3] * m3) / totalMass;
            store.comY[node] = (store.comY[c] * m0 + store.comY[c + 1] * m1 + store.comY[c + 2] * m2 + store.comY[c + 3] * m3) / totalMass;
        }
        // A totalMass of exactly 0 (a quadrant nothing ever landed in) leaves comX/comY at
        // their initialized 0 rather than dividing by zero - harmless, since every reader
        // of a node's aggregate (gravity, energy, range queries) checks mass > 0 first.
    }
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
 * the tree's boundary for structural purposes only (see findQuadrant) - their true
 * position and mass still feed the aggregate mass/center-of-mass math untouched, so the
 * physics stays correct; only which quadrant a stray particle nominally sorts into is
 * affected, and at that distance its exact position barely matters to anyone's force anyway.
 *
 * @returns the module's reusable flat node store; treat as opaque, pass it back into
 *   computeGravity/computePotentialEnergy/findNearbyParticles as `tree`.
 */
export function buildQuadtree(system) {
    store.count = 0;
    store.parentCount = 0;

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

    ensureCapacity(store, 1);
    initNode(0, rootX, rootY, size, -1); // -1: nothing comes after the whole tree
    store.count = 1;

    for (let i = 0; i < system.count; i++) {
        insert(system, i);
    }

    propagate();

    return store;
}

/**
 * Collects every particle index within searchRadius of particle i into `out`, pruning
 * subtrees whose bounding box can't possibly contain a point that close (closest-point-
 * on-box test). Shared by merge.ts and collide.ts, which both need exactly this query and
 * differ only in what they do with the results.
 *
 * Iterative, via the same next-pointer traversal buildQuadtree wires up: skip a pruned or
 * fully-handled subtree by jumping straight to `next` instead of recursing.
 */
export function findNearbyParticles(tree, system, i, searchRadius, out) {
    const px = system.posX[i];
    const py = system.posY[i];
    const searchRadiusSq = searchRadius * searchRadius;
    let node = 0;

    while (node !== -1) {
        if (tree.mass[node] === 0) {
            node = tree.next[node];
            continue;
        }

        const nodeX = tree.x[node];
        const nodeY = tree.y[node];
        const nodeSize = tree.size[node];
        const closestX = Math.max(nodeX, Math.min(px, nodeX + nodeSize));
        const closestY = Math.max(nodeY, Math.min(py, nodeY + nodeSize));
        const dx = px - closestX;
        const dy = py - closestY;

        if (dx * dx + dy * dy > searchRadiusSq) {
            node = tree.next[node]; // whole subtree pruned - its box can't be close enough
            continue;
        }

        if (tree.children[node] === -1) {
            const occ = tree.occupant[node];
            if (occ !== -1) {
                if (occ !== i) out.push(occ);
            } else if (tree.bucket[node]) {
                for (const j of tree.bucket[node]) {
                    if (j !== i) out.push(j);
                }
            }
            node = tree.next[node];
        } else {
            node = tree.children[node]; // box is close enough - check children individually
        }
    }
}
