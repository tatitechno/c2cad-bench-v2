"""Neutral-vocabulary twins of the v2 prompts.

The twin keeps every geometric statement and every number, but replaces object and domain
nouns (staircase, cable, cochlea, vertebra, ...) with neutral part names, so a model cannot build
from a remembered object. Mathematical and geometric terms (tangent, helix, golden angle,
icosahedron, body-centered cubic) are kept: they are the concepts under test, not the object.

Checks (tests/test_neutral.py): no banned domain word survives; the multiset of numbers is
unchanged except for the listed conversions (clock positions become explicit angles).
"""
from __future__ import annotations

import re

# (pattern, replacement) applied in order; patterns are case-insensitive, first-letter case preserved.
SUBS: dict[str, list[tuple[str, str]]] = {
    "Spiral Staircase": [(r"spiral staircase", "helical assembly"), (r"staircase", "assembly"),
                         (r"treads", "element beams"), (r"tread's", "element's"), (r"tread", "element"),
                         (r"cylindrical pillar", "core cylinder"), (r"pillar's", "core cylinder's"), (r"pillar", "core cylinder")],
    "Cannonball Pyramid": [(r"tetrahedral cannonball pyramid", "tetrahedral stack"), (r"cannonball", ""), (r"pyramid", "stack")],
    "Voxel Grid": [],
    "Domino Ring": [(r"equally spaced archways around a circular\nmonument", "equally spaced units around a circular\narrangement"),
                    (r"archways", "units"), (r"monument", "arrangement"), (r"arches", "units"), (r"arch's", "unit's"),
                    (r"arch", "unit"), (r"pillars", "cylinders"), (r"pillar", "cylinder"), (r"beam lintel's", "connecting beam's"),
                    (r"lintel", "connecting beam")],
    "DNA Helix": [(r"base pairs", "node pairs"), (r"backbone locus", "node locus"), (r"backbone node", "node"),
                  (r"sphere nodes", "sphere nodes"), (r"bond", "connector")],
    "Flanged Pipe Joint": [(r"Model a joint with", "Model an assembly with"), (r"joint axis", "assembly axis"),
                           (r"joint gap", "central gap"), (r"mating flanges", "facing rings"), (r"flange's", "ring's"),
                           (r"flanges", "rings"), (r"flange", "ring"), (r"pipe body", "pipe"), (r"bodies", "pipes"),
                           (r"cylinder bolts", "cylinder pins"), (r"bolt circle", "pin circle"), (r"bolt radius", "pin radius"),
                           (r"bolts", "pins"), (r"bolt", "pin"), (r"torus nut", "torus"), (r"nuts", "tori"), (r"nut", "torus")],
    "Suspension Bridge": [(r"symmetric cable bridge", "symmetric beam assembly"), (r"its deck is", "its main member is"),
                          (r"deck centerline", "main-member centerline"), (r"deck midpoint", "main-member midpoint"),
                          (r"deck endpoint", "main-member endpoint"), (r"deck", "main member"),
                          (r"cylinder tower", "cylinder post"), (r"towers", "posts"), (r"tower", "post"),
                          (r"cable beams", "inclined beams"), (r"cables", "inclined beams"), (r"cable", "inclined beam")],
    "Planetary Array": [(r"central sun cylinder", "central cylinder"), (r"sun's", "central cylinder's"),
                        (r"sun-to-planet", "center-to-outer"), (r"sun", "central cylinder"),
                        (r"planet cylinders", "outer cylinders"), (r"planets", "outer cylinders"), (r"planet", "outer cylinder")],
    "Cross-Braced Truss": [(r"tower with", "frame with"), (r"stories", "levels"), (r"story-corner joints", "level-corner points"),
                           (r"story", "level"), (r"beam columns", "beam uprights"), (r"columns", "uprights"),
                           (r"diagonal bracing", "diagonal beams"), (r"braces", "diagonals")],
    "Fractal Y-Tree": [(r"branching beam tree", "branching beam structure"), (r"branching generations", "subdivision generations"),
                       (r"trunk", "root beam"), (r"branch tip", "beam tip"), (r"children", "sub-beams"), (r"child", "sub-beam")],
    "BCC Lattice": [],
    "Ball Bearing Assembly": [(r"open radial bearing with (\d+) rolling\nballs", r"assembly with \1\nspheres"),
                              (r"bearing center", "assembly center"), (r"inner race", "inner ring"), (r"outer race", "outer ring"),
                              (r"raceway surfaces", "ring surfaces"), (r"races", "rings"), (r"balls", "spheres"), (r"ball", "sphere"),
                              (r"symbolic uncaged bearing", "symbolic assembly")],
    "Furniture Assembly": [(r"Model a table with", "Model an assembly with"), (r"cylinder legs", "cylinder supports"),
                           (r"box tabletop", "box slab"), (r"tabletop's", "slab's"), (r"tabletop", "slab"),
                           (r"corner-leg", "corner-support"), (r"leg radius", "support radius"), (r"leg radii", "support radii"),
                           (r"legs", "supports"), (r"leg", "support")],
    "Pipe Manifold": [(r"header manifold with (\d+) forward branches", r"pipe assembly with \1 forward side pipes"),
                      (r"header's", "main pipe's"), (r"header", "main pipe"), (r"back wall", "backing slab"),
                      (r"End caps", "End discs"), (r"cap", "end disc"), (r"Branch junctions", "Side-pipe junctions"),
                      (r"branches", "side pipes"), (r"branch", "side pipe"), (r"coaxial flange", "coaxial collar"),
                      (r"flange", "collar"), (r"cylinder valve body", "outer cylinder"), (r"valve bodies are symbolic embedded solids rather than bored valves",
                                                                               "outer cylinders are symbolic embedded solids"),
                      (r"valve", "outer cylinder"), (r"box bracket", "support box"), (r"bracket", "support box"), (r"wall", "backing slab")],
    "Axle Bearing": [(r"axle assembly", "assembly"), (r"box support block", "box block"), (r"cylinder shaft", "cylinder rod"),
                     (r"shaft", "rod"), (r"Pipe bearings", "Pipe sleeves"), (r"bearings", "sleeves")],
    "Armillary Sphere": [(r"armillary shells", "shells"), (r"equatorial torus girdle", "equatorial torus"), (r"girdles", "equatorial tori")],
    "Clock Tower Mechanism": [(r"clock face with", "disc face with"), (r"positive Z is forward and positive Y is twelve o'clock", "positive Z is forward"),
                              (r"The cylinder back\nplate", "The cylinder back\nplate"), (r"cylinder shaft", "cylinder rod"),
                              (r"shaft axis", "rod axis"), (r"shaft clearance", "rod clearance"), (r"shaft", "rod"),
                              (r"pipe bushing", "short pipe"), (r"dial torus", "face torus"), (r"Dial and markers", "Face torus and markers"),
                              (r"minute hand", "long beam"), (r"hour hand", "short beam"), (r"Each hand", "Each of these two beams"),
                              (r"hand centerline", "beam centerline"), (r"points toward twelve o'clock", "points along positive Y"),
                              (r"points toward eleven o'clock", "points 30 degrees counterclockwise from positive Y as viewed from positive Z"),
                              (r"counterweight cone", "cone"), (r"minute counterweight", "long-beam cone"), (r"hour counterweight", "short-beam cone"),
                              (r"bushing", "short pipe"), (r"dial", "face torus")],
    "Gantry Crane Assembly": [(r"centered gantry crane with (\d+) bays", r"centered frame with \1 spans"), (r"gantry midpoint", "frame midpoint"),
                              (r"bay boundary", "span boundary"), (r"bay length", "span length"), (r"bays", "spans"), (r"bay", "span"),
                              (r"vertical cylinder column", "vertical cylinder"), (r"cylinder column", "vertical cylinder"), (r"column-row", "cylinder-row"), (r"column-axis", "cylinder-axis"),
                              (r"column-top", "cylinder-top"), (r"columns", "vertical cylinders"), (r"column", "vertical cylinder"),
                              (r"beam rails", "top beams"), (r"rail centerlines", "top-beam centerlines"), (r"rail upper faces", "top-beam upper faces"),
                              (r"rails", "top beams"), (r"X-braces", "X-shaped diagonal pairs"), (r"braces", "diagonals"),
                              (r"bridge beam", "cross beam"), (r"bridge", "cross beam"), (r"trolley box", "box"), (r"trolley", "box"),
                              (r"cylinder drum", "cylinder"), (r"drum", "cylinder"), (r"pipe cable", "pipe"), (r"cable", "pipe"),
                              (r"cone hook", "cone"), (r"hook", "cone"), (r"front row", "first row"), (r"is the front", "is the first row"),
                              (r"rail", "top beam"), (r"front", "first row")],
    "Phyllotaxis Disc": [(r"golden-angle phyllotaxis disc", "golden-angle spiral disc"), (r"spherical seeds", "spheres"),
                         (r"seed indices", "sphere indices"), (r"first seed", "first sphere"), (r"seeds", "spheres"),
                         (r"seed radii", "sphere radii"), (r"seed's", "sphere's"), (r"most distant seed", "most distant sphere"),
                         (r"receptacle's", "base cylinder's"), (r"receptacle", "base cylinder")],
    "Compound Eye": [(r"symbolic compound eye", "symbolic dome assembly"), (r"rings\nof ommatidia and a single polar ommatidium", "rings\nof units and a single polar unit"),
                     (r"ommatidia", "units"), (r"ommatidium", "unit"), (r"optical units", "units"), (r"optical parts", "unit parts"),
                     (r"lens's", "sphere's"), (r"lens", "unit sphere"), (r"cylinder guide", "cylinder rod"), (r"guide", "rod"),
                     (r"pipe nerve bundle", "pipe"), (r"nerve", "pipe"), (r"torus mounting ring", "torus"), (r"mounting ring", "torus")],
    "Diatom Frustule": [(r"bilaterally symmetric symbolic diatom frustule", "bilaterally symmetric symbolic shell assembly"),
                        (r"box valves", "box plates"), (r"valves", "plates"), (r"valve", "plate"), (r"sphere nodule", "sphere"),
                        (r"nodule", "central sphere"), (r"raphe", "axial beam"), (r"pipe\nmantle", "pipe\nconnector"),
                        (r"mantle", "connector"), (r"torus band", "torus"), (r"bands", "tori"), (r"band", "torus"),
                        (r"cylinder costae", "cylinder ribs"), (r"costae", "ribs"), (r"costa-side", "rib-side"),
                        (r"sphere pore marker", "marker sphere"), (r"pores", "marker spheres"), (r"Pores", "Marker spheres")],
    "Honeycomb Lattice": [(r"reinforced honeycomb", "reinforced hexagonal cell array")],
    "Vertebral Column": [(r"symbolic cervical and thoracic vertebral column", "symbolic chain of blocks"),
                         (r"symbolic cervical vertebral column", "symbolic chain of blocks"),
                         (r"symbolic thoracic vertebral column", "symbolic chain of blocks"),
                         (r"first vertebral-body center", "first block center"),
                         (r"Z is superior and Y\nanterior", "Z is the up direction and Y\nthe front direction"),
                         (r"Cervical lordosis is (\d+) degrees distributed over (\d+) bodies, bending toward positive Y",
                          r"The total bend is \1 degrees distributed over \2 blocks, bending toward positive Y"),
                         (r"Thoracic kyphosis is (\d+) degrees distributed over (\d+) bodies, bending toward negative Y",
                          r"The total bend is \1 degrees distributed over \2 blocks, bending toward negative Y"),
                         (r"There are (\d+) cervical bodies with (\d+) degrees of lordosis toward positive Y, followed by (\d+) thoracic bodies with (\d+) degrees of kyphosis toward negative Y",
                          r"A first region of \1 blocks bends \2 degrees in total toward positive Y; it is followed by a second region of \3 blocks bending \4 degrees in total toward negative Y"),
                         (r"local superior direction", "local up direction"), (r"superior body's", "upper block's"),
                         (r"posterior pointed cone", "rear pointed cone"), (r"posterior face", "rear face"),
                         (r"posterior direction", "rear direction"), (r"transverse\nbeam processes", "side\nbeams"),
                         (r"positive-X process", "positive-X side beam"), (r"negative-X process", "negative-X side beam"),
                         (r"body boxes", "block boxes"), (r"bodies", "blocks"), (r"body", "block"),
                         (r"disc thickness", "spacer thickness"), (r"discs", "spacer cylinders"), (r"disc", "spacer"),
                         (r"pipe canal segment", "pipe segment"), (r"canal segments", "pipe segments"), (r"canal", "pipe segments"),
                         (r"Boolean spinal solid", "Boolean solid"), (r"posterior", "rear")],
    "Cochlear Spiral": [(r"tapered cochlear helix", "tapered helix"), (r"basal plane", "base plane"), (r"box membrane", "box plate"),
                        (r"membrane pairs", "plate pairs"), (r"Membranes", "Plates"), (r"central cylinder modiolus", "central cylinder"),
                        (r"modiolus", "central cylinder"), (r"basal torus", "base torus"), (r"basal ring", "base torus"),
                        (r"coaxial apex cone", "coaxial top cone"), (r"apex cone", "top cone"), (r"entry pipe", "side pipe")],
    "Radiolarian Skeleton": [(r"radiolarian skeleton", "spherical frame"), (r"outward radial cone", "outward radial cone"),
                             (r"radial spines", "radial cones"), (r"spines", "radial cones")],
}

BANNED: dict[str, list[str]] = {
    "Spiral Staircase": ["stair", "tread", "pillar"],
    "Cannonball Pyramid": ["cannonball", "pyramid"],
    "Domino Ring": ["arch", "monument", "lintel", "pillar", "domino"],
    "DNA Helix": ["DNA", "base pair", "backbone", "bond"],
    "Flanged Pipe Joint": ["flange", "bolt", "nut", "joint"],
    "Suspension Bridge": ["bridge", "deck", "tower", "cable"],
    "Planetary Array": ["sun", "planet"],
    "Cross-Braced Truss": ["tower", "stor", "column", "brac"],
    "Fractal Y-Tree": ["tree", "trunk", "child", "branch tip"],
    "Ball Bearing Assembly": ["bearing", "race", "ball"],
    "Furniture Assembly": ["table", "leg"],
    "Pipe Manifold": ["manifold", "header", "wall", "cap", "flange", "valve", "bracket", "branch"],
    "Axle Bearing": ["axle", "shaft", "bearing"],
    "Armillary Sphere": ["armillary", "girdle"],
    "Clock Tower Mechanism": ["clock", "o'clock", "shaft", "bushing", "dial", "hand", "minute", "hour", "counterweight"],
    "Gantry Crane Assembly": ["gantry", "crane", "bay", "column", "rail", "brace", "bridge", "trolley", "drum", "cable", "hook", "front"],
    "Phyllotaxis Disc": ["phyllotaxis", "seed", "receptacle"],
    "Compound Eye": ["eye", "ommatidi", "lens", "nerve", "optical", "guide", "mounting"],
    "Diatom Frustule": ["diatom", "frustule", "valve", "nodule", "raphe", "mantle", "band", "costa", "pore"],
    "Honeycomb Lattice": ["honeycomb"],
    "Vertebral Column": ["vertebra", "cervical", "thoracic", "lordosis", "kyphosis", "superior", "anterior", "posterior",
                         "process", "disc", "canal", "spinal", "body", "bodies"],
    "Cochlear Spiral": ["cochlea", "modiolus", "membrane", "basal", "apex", "entry"],
    "Radiolarian Skeleton": ["radiolarian", "skeleton", "spine"],
}

# numbers a twin may add (explicit geometric statement of a domain convention)
ADDED_NUMBERS = {"Clock Tower Mechanism": ["30"]}


def _case_preserving(repl: str):
    def f(m):
        r = m.expand(repl)
        s = m.group(0)
        if s[:1].isupper() and r[:1].islower():
            r = r[:1].upper() + r[1:]
        return r
    return f


def twin(family: str, body: str) -> str:
    out = body
    for pat, rep in SUBS.get(family, []):
        pat = pat.replace("\n", " ").replace(" ", r"\s+")          # phrases may wrap across lines
        out = re.sub(r"(?<![A-Za-z])" + pat + r"(?![A-Za-z])", _case_preserving(rep.replace("\n", " ")), out, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", out)


def numbers(text: str) -> list[str]:
    return sorted(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text))


def violations(family: str, text: str) -> list[str]:
    return [w for w in BANNED.get(family, []) if re.search(r"(?<![A-Za-z])" + re.escape(w), text, flags=re.IGNORECASE)]
