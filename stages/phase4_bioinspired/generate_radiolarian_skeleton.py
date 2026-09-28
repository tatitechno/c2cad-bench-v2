"""
Radiolarian Skeleton — Phase 4 Bio-Inspired
=============================================
Models a radiolarian (marine microorganism) siliceous skeleton: a
geodesic sphere built from interconnected struts with radial spines.

WHY THIS IS HARD FOR LLMs:
  This family is designed to be near-impossible for current LLMs because
  it compounds ALL FIVE identified failure mechanisms:

  1. CONTINUOUS 3D ANGULAR COMPUTATION — Every vertex of the geodesic
     sphere sits at a unique (θ, φ) on the sphere. The model must compute
     spherical-to-Cartesian conversions for each vertex. Unlike Spiral
     Staircase (2D angles only), this requires full 3D trigonometry.

  2. BEAM ORIENTATION BLINDNESS — The skeleton is 100% beam-based for
     struts and spines. Every strut connects two geodesic vertices, so
     every beam has a unique 3D direction vector. With 30 struts at L1,
     this produces 30 unique orientations — far beyond the 10 unique
     orientations that brought Spiral Staircase to 60%.

  3. MULTI-TYPE COMPOSITION — Uses 4 types (beam, sphere, cone, torus).
     Spheres mark vertices, beams are struts, cones are radial spines,
     and a torus marks the equator.

  4. COMBINATORIAL CONNECTIVITY — The model must figure out which vertex
     pairs are connected by struts. This is graph reasoning: given the
     icosahedral subdivision, determine the edge list. LLMs have no
     spatial working memory to track this.

  5. TYPE-SELECTIVE UNDER-PRODUCTION — Cones (spines) are the hardest
     type (54.4% accuracy). With cones at every vertex pointing radially
     outward, the model must compute a unique cone orientation per vertex.

The skeleton consists of:
  • Vertex nodes: spheres at each geodesic vertex
  • Struts: beams connecting adjacent vertices
  • Radial spines: cones at each vertex pointing outward from centre
  • Equatorial ring: torus at the equator

The geometry is based on an icosahedron:
  L1: Icosahedron (12 vertices, 30 edges)        → 12 spheres + 30 beams + 12 cones + 1 torus = 55 shapes
  L2: 1× subdivision (42 vertices, 120 edges)     → 42 + 120 + 42 + 1 = 205 shapes
  L3: 2× subdivision (162 vertices, 480 edges)    → 162 + 480 + 162 + 1 = 805 shapes
"""

import math


def _icosahedron_vertices(R):
    """Return the 12 vertices of a regular icosahedron of circumradius R."""
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    norm = math.sqrt(1 + phi ** 2)
    a = R / norm
    b = R * phi / norm
    verts = [
        ( 0,  a,  b), ( 0,  a, -b), ( 0, -a,  b), ( 0, -a, -b),
        ( a,  b,  0), ( a, -b,  0), (-a,  b,  0), (-a, -b,  0),
        ( b,  0,  a), ( b,  0, -a), (-b,  0,  a), (-b,  0, -a),
    ]
    return [(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in verts]


def _icosahedron_edges():
    """Return the 30 edges of a regular icosahedron as (i, j) pairs."""
    # Precomputed adjacency for a standard icosahedron
    edges = [
        (0, 2), (0, 4), (0, 6), (0, 8), (0, 10),
        (1, 3), (1, 4), (1, 6), (1, 9), (1, 11),
        (2, 5), (2, 7), (2, 8), (2, 10),
        (3, 5), (3, 7), (3, 9), (3, 11),
        (4, 6), (4, 8), (4, 9),
        (5, 7), (5, 8), (5, 9),
        (6, 10), (6, 11),
        (7, 10), (7, 11),
        (8, 9),
        (10, 11),
    ]
    return edges


def _subdivide(vertices, edges, R):
    """Subdivide each edge by inserting a midpoint, project to sphere, return new vertices+edges."""
    # Map edges to midpoints
    mid_cache = {}
    new_verts = list(vertices)
    new_edges = []

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in mid_cache:
            return mid_cache[key]
        mx = (new_verts[i][0] + new_verts[j][0]) / 2.0
        my = (new_verts[i][1] + new_verts[j][1]) / 2.0
        mz = (new_verts[i][2] + new_verts[j][2]) / 2.0
        # Project onto sphere
        norm = math.sqrt(mx**2 + my**2 + mz**2)
        if norm > 0:
            mx, my, mz = mx*R/norm, my*R/norm, mz*R/norm
        idx = len(new_verts)
        new_verts.append((round(mx, 4), round(my, 4), round(mz, 4)))
        mid_cache[key] = idx
        return idx

    # Build face list from edges (reconstruct triangles)
    # For an icosahedron, we know there are 20 faces
    # For subsequent subdivisions, we reconstruct faces from edges
    # Simpler approach: for each edge, split into 2
    for i, j in edges:
        m = get_midpoint(i, j)
        new_edges.append((i, m))
        new_edges.append((m, j))

    # Now add the "lateral" edges within each original face
    # For each triangle face, connect the 3 midpoints
    # Reconstruct faces from original edges
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # Find triangles: (a, b, c) where a-b, b-c, a-c are all edges
    faces = set()
    for a in range(len(vertices)):
        for b in adj[a]:
            if b <= a: continue
            for c in adj[a] & adj[b]:
                if c <= b: continue
                faces.add((a, b, c))

    for a, b, c in faces:
        ma = get_midpoint(b, c)
        mb = get_midpoint(a, c)
        mc = get_midpoint(a, b)
        new_edges.append((mc, ma))
        new_edges.append((ma, mb))
        new_edges.append((mb, mc))

    # Deduplicate edges
    edge_set = set()
    for i, j in new_edges:
        edge_set.add((min(i, j), max(i, j)))

    return new_verts, sorted(edge_set)


def _reference_solution(scale):
    """Build the canonical radiolarian skeleton."""
    R = 50.0  # mm — sphere radius
    shapes = []
    sid = 0

    # Build geodesic mesh
    vertices = _icosahedron_vertices(R)
    edges = _icosahedron_edges()

    # Subdivide based on scale
    if scale == 1:
        pass  # Base icosahedron
    elif scale == 2:
        vertices, edges = _subdivide(vertices, edges, R)
    elif scale == 3:
        vertices, edges = _subdivide(vertices, edges, R)
        vertices, edges = _subdivide(vertices, edges, R)

    # Parameters
    node_r = 2.0          # mm — vertex node sphere radius
    strut_w = 1.5         # mm — strut beam width
    spine_h = 12.0        # mm — radial spine cone height
    spine_r_base = 2.5    # mm — spine cone base radius
    spine_r_tip = 0.3     # mm — spine cone tip radius
    torus_major = R       # mm — equatorial ring major radius
    torus_minor = 1.5     # mm — equatorial ring tube radius

    # ── Equatorial ring (torus) ──
    shapes.append({
        "id": sid, "type": "torus",
        "center": [0.0, 0.0, 0.0],
        "major_radius": torus_major,
        "minor_radius": torus_minor,
    })
    sid += 1

    # ── Vertex nodes (spheres) ──
    for v in vertices:
        shapes.append({
            "id": sid, "type": "sphere",
            "center": [round(v[0], 2), round(v[1], 2), round(v[2], 2)],
            "radius": node_r,
        })
        sid += 1

    # ── Struts (beams connecting adjacent vertices) ──
    for i, j in edges:
        vi, vj = vertices[i], vertices[j]
        shapes.append({
            "id": sid, "type": "beam",
            "start": [round(vi[0], 2), round(vi[1], 2), round(vi[2], 2)],
            "end":   [round(vj[0], 2), round(vj[1], 2), round(vj[2], 2)],
            "width": strut_w,
            "height": strut_w,
        })
        sid += 1

    # ── Radial spines (cones pointing outward from each vertex) ──
    for v in vertices:
        # Direction: from origin through vertex, normalised
        norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if norm == 0:
            continue
        dx, dy, dz = v[0]/norm, v[1]/norm, v[2]/norm
        # Cone base is at the vertex, tip extends outward
        base_x = v[0]
        base_y = v[1]
        base_z = v[2]
        tip_x = v[0] + spine_h * dx
        tip_y = v[1] + spine_h * dy
        tip_z = v[2] + spine_h * dz
        # Centre of cone is midpoint
        cx = round((base_x + tip_x) / 2.0, 2)
        cy = round((base_y + tip_y) / 2.0, 2)
        cz = round((base_z + tip_z) / 2.0, 2)
        shapes.append({
            "id": sid, "type": "cone",
            "center": [cx, cy, cz],
            "base_radius": spine_r_base,
            "top_radius": spine_r_tip,
            "height": spine_h,
            "axis": [round(dx, 4), round(dy, 4), round(dz, 4)],
        })
        sid += 1

    return shapes


def generate_radiolarian(scale):
    """
    Radiolarian Skeleton (Phase 4 Bio-Inspired).
    Scale = subdivision level (1=icosahedron, 2=1× subdiv, 3=2× subdiv).
    """
    ref = _reference_solution(scale)
    total = len(ref)

    R = 50.0
    node_r = 2.0
    strut_w = 1.5
    spine_h = 12.0
    spine_r_base = 2.5
    spine_r_tip = 0.3
    torus_major = R
    torus_minor = 1.5

    # Count components
    if scale == 1:
        n_verts, n_edges = 12, 30
    elif scale == 2:
        n_verts, n_edges = 42, 120
    else:
        n_verts, n_edges = 162, 480

    prompt = f"""Model a radiolarian skeleton with a spherical
geodesic cage of circumradius 50 and {scale - 1} recursive spherical midpoint
refinement levels of its regular icosahedral triangulation. Use the conventional
four-subtriangle refinement at each level. Center the regular icosahedron at the origin. Its vertices
are the corners of three mutually perpendicular golden rectangles centered at
the origin, lying in the YZ, XY, and XZ planes. Their long edges are parallel
to Z, Y, and X respectively. This fixes the orientation without a vertex list.
Each mesh vertex has a sphere node of radius 2. Adjacent vertex centers are
joined by beam struts of square section 1.5, one per triangular-mesh edge.
At each vertex an outward radial cone has its base-face center on the vertex
locus, base radius 2.5, tip radius 0.3 and length 12. A torus follows the
equator of the cage sphere, with tube radius 1.5. ID 0 is the equatorial ring;
then index vertex nodes, struts, radial spines. Node ordering may be any
deterministic ordering, with spines in the corresponding node order.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    specs = {
        "gravity_check": False,
        "interference_check": True,
        "mates": [],
        "clearance_fit": [],
        "reference": ref
    }
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    prompt, specs = generate_radiolarian(args.scale)
    print(f"Radiolarian — scale {args.scale}, {len(specs['reference'])} shapes")
    print(prompt)
