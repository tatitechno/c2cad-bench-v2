"""Traceability: prompt sentence <-> constraint clause keys.

TRACE[family] is a list of (anchor phrase, keys). A sentence is covered when it contains an anchor.
keys are clause keys used by the family's builder, or one of the tags
  ids        ordering rule, evaluated by id binding (semantic_id_binding), not by geometry
  symbolic   declares parts as symbolic/embedded (affects interference and volume metrics only)
  note       restates or clarifies another sentence; carries no separate geometric requirement
Tests require: every sentence of every case is covered; every builder key is traced to a sentence.
"""
from __future__ import annotations

import re

TRACE = {
    "Spiral Staircase": [
        ("Model a spiral staircase with", ["types"]), ("pillar axis passes through the origin", ["pillar"]),
        ("cylindrical pillar has radius", ["pillar"]), ("Each tread has radial length", ["tread_dims"]),
        ("Adjacent tread bottom faces differ", ["rise"]), ("treads per complete revolution", ["turn"]),
        ("first tread points along positive X", ["first"]), ("inner centerline endpoint meets", ["inner", "radial"]),
        ("ends one rise above", ["pillar_top", "pillar"]), ("ID 0 is the pillar", ["ids"])],
    "Cannonball Pyramid": [
        ("tetrahedral cannonball pyramid", ["types", "radius", "tangent"]), ("lowest layer rests on", ["ground"]),
        ("vertical projection of one corner sphere", ["corner", "edge"]), ("triangular pockets", ["tangent", "no_overlap"]),
        ("pocket orientation toward the interior", ["tangent"]), ("Index from the bottom upward", ["ids"])],
    "Voxel Grid": [
        ("cubic grid with", ["types", "aligned"]), ("Each box is a cube of side", ["cube"]),
        ("Facing surfaces of neighboring boxes", ["gap"]), ("minimum outer corner", ["corner"]), ("Index X fastest", ["ids"])],
    "Domino Ring": [
        ("equally spaced archways", ["types", "arches"]), ("Pillar axes lie on a circle", ["circle"]),
        ("symmetric pair of vertical cylinder pillars", ["pillar", "pair"]), ("Both pillars stand on the ground", ["ground"]),
        ("lintel's centerline joins", ["lintel_ends"]), ("width equals a pillar diameter", ["lintel_dims"]),
        ("first arch's radial bisector", ["first", "arches"]), ("Within each arch index", ["ids"])],
    "DNA Helix": [
        ("right-handed double helix", ["types", "turn"]), ("backbone locus has radius", ["locus"]),
        ("diametrically opposite sphere nodes", ["dims", "opposite", "bond"]), ("first pair lies in the XY plane", ["first"]),
        ("Successive levels rise by", ["rise", "turn"]), ("Index upward", ["ids"])],
    "Flanged Pipe Joint": [
        ("Model a joint with", ["types"]), ("joint axis is X", ["axis", "gap"]), ("On each side is a pipe body", ["dims", "abut"]),
        ("Each flange is a pipe", ["dims"]), ("facing flanges have an axial gap", ["gap", "abut"]),
        ("Cylinder bolts have radius", ["dims", "axis", "bolt_center", "bolt_circle", "spacing"]),
        ("first bolt lies on positive Y", ["first", "spacing"]), ("Each bolt has a torus nut", ["nut_dims", "nut_plane", "nut_coax"]),
        ("Nuts and bolts are symbolic", ["symbolic"]), ("IDs 0 and 1 are", ["ids"]), ("IDs 2 and 3 are", ["ids"]),
        ("Then index alternating bolt", ["ids"])],
    "Suspension Bridge": [
        ("symmetric cable bridge", ["types", "mirror"]), ("deck is a beam of span", ["deck", "section"]),
        ("deck midpoint projects to the origin", ["deck"]), ("At each deck endpoint", ["tower", "tower_at_end"]),
        ("cable beams of square section", ["section", "cable_top", "cable_on_deck", "uniform"]),
        ("outermost attachment", ["outermost", "innermost"]), ("Include both limiting attachments", ["outermost", "innermost"]),
        ("Index deck", ["ids"])],
    "Planetary Array": [
        ("central sun cylinder of radius", ["types", "dims"]), ("sun center is the origin", ["sun", "axes"]),
        ("All centers lie in the XY plane", ["plane"]), ("externally tangent", ["tangent"]),
        ("equally spaced angularly", ["first", "spacing"]), ("Index the sun before", ["ids"]),
        ("Only sun-to-planet tangency is prescribed", ["note"])],
    "Cross-Braced Truss": [
        ("Model a tower with", ["types", "column", "corner"]), ("footprint midpoint is the origin", ["corner", "story"]),
        ("vertical beam columns at its corners", ["column", "joint"]), ("Beam ends coincide", ["joint", "story"]),
        ("Every beam has square section", ["section"]), ("Index stories upward", ["ids"])],
    "Fractal Y-Tree": [
        ("binary branching beam tree", ["types", "plane"]), ("trunk has length", ["trunk"]),
        ("All beams have square section", ["section"]), ("children meet their parent's endpoint", ["attach", "half", "turn"]),
        ("For IDs use depth-first order", ["ids"])],
    "BCC Lattice": [
        ("body-centered cubic lattice with", ["types", "lattice"]), ("Unit-cell side length", ["lattice", "strut_diag"]),
        ("minimum lattice corner is the origin", ["corner", "extent"]), ("Corner and body nodes are spheres", ["dims", "unique"]),
        ("Beam struts of square section", ["dims", "strut_nodes", "strut_diag"]), ("Include only these BCC links", ["strut_diag"]),
        ("IDs may follow any", ["ids"])],
    "Ball Bearing Assembly": [
        ("open radial bearing with", ["types", "dims"]), ("bearing center is the origin", ["center", "axis"]),
        ("inner race is a pipe", ["dims"]), ("axial width is", ["dims"]), ("outer race has outer radius", ["dims", "tangent"]),
        ("midplanes in the XY plane", ["center", "plane"]), ("Ball centers lie in that plane", ["plane", "first", "spacing"]),
        ("IDs 0 and 1 are inner", ["ids"]), ("symbolic uncaged bearing", ["symbolic"])],
    "Furniture Assembly": [
        ("Model a table with", ["types", "legs", "top"]), ("origin is on the ground below", ["top"]), ("Z is up", ["note"]),
        ("Legs stand vertically", ["legs", "ground", "touch"]), ("corner-leg axes are inset", ["inset"]),
        ("Extra legs, if needed", ["mirror", "edge_line"]), ("Their stations divide", ["stations"]),
        ("positive-X boundary is", ["stations"]), ("ID 0 is the tabletop", ["ids"]), ("Index corner legs", ["ids"]),
        ("Index extra pairs", ["ids"])],
    "Pipe Manifold": [
        ("header manifold with", ["types"]), ("header is a pipe parallel to X", ["header"]), ("inner and outer radii are", ["header"]),
        ("Branch junctions divide", ["junctions"]), ("back wall is an axis-aligned box", ["wall"]), ("side margins extend", ["wall"]),
        ("End caps are cylinders", ["caps"]), ("Branches point along positive Y", ["branch"]), ("Each pipe has inner radius", ["branch"]),
        ("coaxial flange starts", ["flange"]), ("cylinder valve body", ["valve"]), ("box bracket under each junction", ["bracket"]),
        ("valve bodies are symbolic", ["symbolic"]), ("IDs: header 0", ["ids"])],
    "Axle Bearing": [
        ("axle assembly with radial block-bore clearance", ["types", "bore"]), ("box support block has", ["block"]),
        ("stands on the XY ground plane", ["block"]), ("horizontal bore passes", ["bore"]), ("cut envelope as cylinder", ["bore"]),
        ("centered cylinder shaft", ["shaft"]), ("Derive the bore radius", ["bore"]), ("Pipe bearings each have", ["bearing"]),
        ("Bearings sit outside the block", ["abut", "bearing"]), ("ID 0 is the block", ["ids"]), ("IDs 3 and 4 are", ["ids"]),
        ("bore cylinder is a symbolic cut envelope", ["symbolic"])],
    "Armillary Sphere": [
        ("concentric armillary shells", ["types", "rings", "girdle"]), ("innermost radius is", ["rings", "girdle", "rod_ends"]),
        ("great-circle rings", ["rings", "girdle"]), ("cylinder rods of radius", ["rod_dims", "rod_dir", "rod_ends", "rod_unique"]),
        ("Center the regular icosahedron", ["rod_dir"]), ("golden rectangles", ["rod_dir"]), ("long edges are parallel", ["rod_dir"]),
        ("fixes the orientation without a vertex list", ["note"]), ("IDs: great-circle rings", ["ids"]),
        ("any consistent ordering", ["ids"])],
    "Clock Tower Mechanism": [
        ("clock face with", ["types", "spacing"]), ("origin is the center of the front face", ["plate", "minute"]),
        ("cylinder back plate", ["plate"]), ("cylinder shaft has radius", ["shaft"]), ("concentric pipe bushing", ["bushing"]),
        ("dial torus of ring radius", ["dial"]), ("Each marker is a radial beam", ["markers"]), ("Its centerline lies in the face plane", ["markers"]),
        ("first marker points along positive X", ["first", "spacing"]), ("minute hand has length", ["minute"]),
        ("hour hand has length", ["hour"]), ("coaxial counterweight cone", ["counterweight"]), ("base-face center is 1 from", ["counterweight"]),
        ("IDs: plate 0", ["ids"]), ("Dial and markers are symbolic", ["symbolic"])],
    "Gantry Crane Assembly": [
        ("centered gantry crane with", ["types", "grid"]), ("ground projection of the gantry midpoint", ["center"]),
        ("At every bay boundary", ["columns", "ground", "grid"]), ("beam rails run along each row", ["rails"]),
        ("negative-Y row is the front", ["braces"]), ("bridge beam spans", ["bridge"]), ("trolley box of X length", ["trolley"]),
        ("cylinder drum", ["drum"]), ("vertical pipe cable", ["cable"]), ("cone hook", ["hook"]), ("Index columns by increasing X", ["ids"]),
        ("Crossed centerlines are symbolic", ["symbolic"])],
    "Phyllotaxis Disc": [
        ("golden-angle phyllotaxis disc", ["types", "seed", "angle"]), ("Fermat spiral with radial scale", ["radius_law"]),
        ("zero-angle ray is positive X", ["first", "angle"]), ("receptacle is a cylinder of thickness", ["receptacle"]),
        ("two seed radii beyond", ["receptacle"]), ("lowest points in the receptacle's midplane", ["embed"]),
        ("ID 0 is the receptacle", ["ids"])],
    "Compound Eye": [
        ("symbolic compound eye with", ["types"]), ("supporting sphere has radius", ["support"]),
        ("divide the polar-angle range", ["polar"]), ("first nonpolar ring contains", ["azimuth"]),
        ("equally spaced in azimuth", ["first", "azimuth"]), ("Each lens is a sphere", ["lens", "locus"]),
        ("radially inward cone", ["cone"]), ("coaxial cylinder guide", ["guide"]), ("pipe nerve bundle", ["nerve"]),
        ("torus mounting ring", ["ring"]), ("symbolic dome carrier", ["symbolic"]), ("IDs 0, 1, 2 are", ["ids"])],
    "Diatom Frustule": [
        ("symbolic diatom frustule", ["types"]), ("origin is midway between the valves", ["gap"]),
        ("X is longitudinal", ["raphe", "costae"]), ("Top and bottom box valves", ["valves", "gap"]), ("sphere nodule of radius", ["nodule"]),
        ("raphe is a longitudinal beam", ["raphe"]), ("central pipe mantle", ["mantle"]), ("torus band at each mantle end", ["bands"]),
        ("equally spaced rib stations", ["stations"]), ("Exclude the center station", ["stations"]),
        ("paired cylinder costae", ["costae"]), ("Their axes and the raphe centerline", ["costae", "raphe"]),
        ("sphere pore marker", ["pores"]), ("IDs: nodule 0", ["ids"]), ("For both repeat groups", ["ids"]),
        ("Pores and ribs are embedded symbolic", ["symbolic"])],
    "Honeycomb Lattice": [
        ("reinforced honeycomb with", ["types", "lattice"]), ("Pipe cells have inner radius", ["cells"]),
        ("planar separation of nearest cell axes", ["lattice"]), ("lattice neighbor of the central cell", ["lattice"]),
        ("origin is the center of a horizontal cylinder base plate", ["base"]), ("base thickness is", ["base"]),
        ("central cell stands on the base plate", ["plane"]), ("All pipe centroids lie", ["plane"]),
        ("Noncentral axes tilt", ["tilt"]), ("Each cell has a cylinder cap", ["caps", "cones"]),
        ("Cap and cone centroids project", ["caps", "cones"]), ("Caps share the horizontal centroid plane", ["caps"]),
        ("Cones share the horizontal centroid plane", ["cones"]), ("nearest-neighbor pair of cell centers", ["links"]),
        ("torus frame at the pipe-centroid plane", ["frame"]), ("uniformly spaced radial beam ribs", ["ribs"]),
        ("first rib points along positive X", ["ribs"]), ("frame and links are symbolic", ["symbolic"]),
        ("IDs: base 0", ["ids"]), ("Within each cell group start at the center", ["ids", "lattice"]),
        ("Link IDs follow", ["ids"])],
    "Vertebral Column": [
        ("vertebral column in the YZ", ["types"]), ("first vertebral-body center is the origin", ["first"]),
        ("Z is superior", ["curve", "cone"]), ("Bodies are axis-aligned boxes", ["body"]), ("Discs have axial thickness", ["disc"]),
        ("lordosis", ["curve"]), ("kyphosis", ["curve"]), ("first local superior direction", ["curve"]),
        ("Equal tilt increments", ["curve"]), ("Regional totals are measured", ["curve", "cone"]),
        ("At the region transition", ["curve"]), ("center-to-center pitch", ["curve"]),
        ("Each superior body's center lies", ["curve"]), ("posterior pointed cone", ["cone"]),
        ("Body boxes themselves remain axis-aligned", ["body"]), ("transverse beam processes", ["process"]),
        ("Disc centers are midway", ["disc"]), ("pipe canal segment", ["canal"]), ("IDs: body, posterior cone", ["ids"]),
        ("Discs and canal are symbolic", ["symbolic"])],
    "Cochlear Spiral": [
        ("tapered cochlear helix with", ["types", "turn"]), ("Its start is on positive X", ["start", "turn", "rise"]),
        ("starting radius is", ["start", "taper"]), ("straight beam segments per revolution", ["segment", "turn"]),
        ("Consecutive endpoints lie on the helix", ["continuity", "turn"]), ("box membrane", ["membrane"]),
        ("central cylinder modiolus", ["modiolus"]), ("basal torus", ["ring"]), ("entry pipe of bore radius", ["entry"]),
        ("coaxial apex cone", ["apex"]), ("IDs: modiolus 0", ["ids"]), ("Membranes are symbolic", ["symbolic"])],
    "Radiolarian Skeleton": [
        ("radiolarian skeleton with a spherical geodesic cage", ["types", "vertex", "ring"]),
        ("conventional four-subtriangle refinement", ["vertex", "edge"]), ("Center the regular icosahedron", ["vertex"]),
        ("golden rectangles", ["vertex"]), ("long edges are parallel", ["vertex"]),
        ("fixes the orientation without a vertex list", ["note"]), ("Each mesh vertex has a sphere node", ["nodes", "unique"]),
        ("beam struts of square section", ["struts", "edge"]), ("outward radial cone", ["spines"]),
        ("torus follows the equator", ["ring"]), ("ID 0 is the equatorial ring", ["ids"]), ("Node ordering may be any", ["ids"])],
}
TAGS = {"ids", "symbolic", "note"}


def sentences(prompt_body: str) -> list[str]:
    body = re.sub(r"\s+", " ", prompt_body).strip()
    return [s for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", body) if s]


def _norm(s):
    return re.sub(r"\s+", " ", s).lower()


def sentence_map(case: dict) -> list[tuple[str, set]]:
    """[(sentence, set of keys/tags covering it)]"""
    out = []
    for s in sentences(case["prompt_body"]):
        keys = set()
        for anchor, ks in TRACE[case["family"]]:
            if _norm(anchor) in _norm(s):
                keys |= set(ks)
        out.append((s, keys))
    return out


def clause_to_sentences(case: dict) -> dict:
    """clause key -> list of prompt sentences that state it."""
    d = {}
    for s, keys in sentence_map(case):
        for k in keys - TAGS:
            d.setdefault(k, []).append(s)
    return d
