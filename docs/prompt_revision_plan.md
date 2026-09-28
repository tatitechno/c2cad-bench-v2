# Plan to revise all benchmark prompts

Scope: the 25 families registered in `runners/run_unified.py`, at all three difficulty levels (75 cases). This plan follows `prompt_design_rules.md`. It does not yet change prompts, references, scorers, or released results.

## Compliance contract

1. Describe physical relationships: tangency, contact, concentricity, support, adjacency, symmetry, and clearance.
2. Give independent inputs only. Component counts, dimensions, pitch constants, clearances, ratios, and defining design angles are permitted inputs. Omit derived coordinates, totals, angular increments, intermediate radii, and evaluated dimensions.
3. Define origin, axes, handedness, first-element orientation, and indexing where needed. Anchors are necessary specification, not automatically answer leakage.
4. Use domain concepts without worked solutions. Do not supply trigonometric conversions, coordinate recurrences, vector normalization instructions, neighbor-walk algorithms, or explicit edge/vertex answer lists. Distinguish a defining relationship (such as branching ratio) from instructions for calculating its coordinates.
5. Use the canonical seven primitives and centroid/endpoint conventions. Include a shared schema/output contract without example assemblies or answer coordinates. Preserve component IDs when semantic validators depend on them.

For each family, record each numeric input, its role, the physical relationships, the frame, and the geometric quantities the model must derive. Review interpolated values as well as literals: a number produced by an f-string can still leak an answer.

## Release structure

- Preserve the v1 cases, model outputs, scores, summaries, and viewer database as the evaluated snapshot.
- Introduce a named strict prompt version with a separate 75-case export, audit, and result destination. The runner must select that version explicitly and must not reuse v1 results as strict-version results.
- Keep family/difficulty correspondence, with prompt version and prompt hash recorded in the evaluation manifest/results metadata.
- Initially retain existing golden geometry and scoring. Where references contradict the intended physical specification, document the conflict and separate any reference correction into an explicit benchmark revision. Do not distort the new wording to conceal a reference defect.
- Inspect generator return values and runtime prompt suffixes so the model receives only the selected prompt and the shared output contract.

## Family-by-family work

| Phase | Family | Revision and required review |
| --- | --- | --- |
| P1 | Spiral Staircase | Remove total shape count and replace the angular increment with steps per revolution. Keep step count, rise, dimensions, contact, ground, and first-step orientation. Define pillar extent so its height is derived uniquely; the current phrase “accommodate all steps” is insufficient to reproduce its height. |
| P1 | Cannonball Pyramid | Remove total spheres, the stated base-centroid height, and doubled-radius neighbor distance. Specify tangent close packing on the ground, layer count, base orientation, and which triangular pockets support upper layers. |
| P1 | Voxel Grid | Remove the evaluated cube count. Retain cells per axis, cube size, surface gap, minimum outer corner at origin, positive-axis extent, and indexing order. |
| P1 | Domino Ring | Remove total parts, computed inter-arch spacing, and angle-offset placement instructions. Describe equally spaced archways, pillar separation, beam support, ring radius, and first arch. Identify independent opening/section dimensions and verify their relationship to the current angular offsets and beam endpoints. |
| P1 | DNA Helix | Remove total parts and coordinate-language placement instructions. Retain pair count, radius, levels per revolution, rise, handedness, opposite pairing, first-pair frame, and bonds joining nodes. |
| P1 | Flanged Pipe Joint | Remove angular increments, sine/cosine instructions, and total shapes. Retain bolt count, independent sizes, concentric alignment, flange gap, and physical bolt/nut placement. Derive dependent bolt-circle and bolt-length quantities where applicable. Anchor the first bolt and verify that contact/interference wording matches the reference. |
| P2 | Suspension Bridge | Replace tower/deck coordinates and cable attachment endpoints with deck span/elevation, tower height, symmetry, and physical attachment margins. Keep cables per side and uniform attachment spacing. Anchor the deck midpoint and axes. |
| P2 | Planetary Array | Retain planet count, component sizes, tangency to the sun, uniform distribution, and first-planet direction. Express origin/axes in plain language; do not add pitch radius or angular increments. Do not introduce planet-to-planet noninterference: high-count current cases may not satisfy it. |
| P2 | Cross-Braced Truss | Replace explicit corner coordinates with square footprint dimensions centered on the vertical axis. Specify story count/height, corner columns, and X-bracing as topology. Remove redundant component counts. |
| P2 | Fractal Y-Tree | Replace trunk endpoint vectors with grounded trunk length/direction. Retain depth, branch-length ratio, branch angle, branching plane, symmetry, and tip attachment as defining design inputs. Remove algorithmic instructions and endpoint equations. |
| P2 | BCC Lattice | Retain BCC unit-cell topology, cell size, grid dimensions, shared nodes, and center-to-corner struts. Remove coordinate vectors, construction sequencing, and redundant per-cell beam totals. Describe shared nodes once. |
| P2 | Ball Bearing Assembly | Remove ball-coordinate formulas, angular increments, computed pitch radius, and total count. Specify rolling-element count, concentric races, contact/clearance relationships, independent dimensions, and first-ball orientation. Check which existing radius values are independent and whether tangency is actually satisfied. |
| P3 | Furniture Assembly | Replace free dimensional choice with the base dimensions required by coordinate scoring. Specify tabletop orientation, ground support, corner insets, and the distribution of extra legs through physical relationships. Check the extra-leg spacing in the reference before claiming equal spacing. Retain required IDs. |
| P3 | Pipe Manifold | Remove evaluated totals and derived header length/spacing redundancy. Supply omitted independent dimensions and sufficient mounting/branch placement constraints. Describe caps, flange clearance, valves, and supports physically. Retain role IDs and anchor the header midpoint/frame. |
| P3 | Axle Bearing | Replace free dimensional ratios with independent block/shaft/bearing dimensions, face relationships, and overhangs. Derive bore radius from radial clearance. Anchor the block and shaft axis. Resolve the requested subtraction operation with the primitive-only output contract and evaluator expectations before finalizing the prompt. |
| P3 | Armillary Sphere | Remove radius lists, evaluated counts, explicit icosahedral vectors, normalization, and rod-center/height equations. Keep base radius, shell scaling ratio, shell count, three orthogonal ring planes, and rods connecting shells along icosahedral directions. Define the icosahedron orientation geometrically. |
| P3 | Clock Tower Mechanism | Remove centroid coordinates, angular increments, trigonometric instructions, and evaluated totals. Describe flush faces, shaft/bushing projection, equally spaced dial markers, hand directions, and counterweight contact. Choose independent dimensions and clearance inputs. Verify the current “10 o’clock” versus “120 degrees” inconsistency; anchor marker zero independently of hand directions. |
| P3 | Gantry Crane Assembly | Remove computed column/brace/total counts. Retain bay count, span, bay length, part dimensions, ground support, X-bracing, and trolley/hoist contacts. Specify centered layout, front row, and all beam sections. Check trolley/bridge support against the golden. |
| P4 | Phyllotaxis Disc | Remove the numerical golden angle, angle/radius equations, Cartesian equations, and seed-centroid height. Use golden-angle phyllotaxis, Fermat spacing with its independent scale, index origin, handedness, and first-seed orientation. Specify receptacle margin uniquely. Resolve the reference’s seed/disc contact mismatch: seed centers at height 3 with a disc centered at zero and thickness 1 are not tangent to the disc top. |
| P4 | Compound Eye | Remove computed ommatidium/shape totals, spherical-coordinate formulas, and nerve centroid coordinates. Retain ring count, independently defined per-ring population pattern, dome radius, angular extent, part sizes, radial orientation, and inward contact relationships. Anchor ring azimuth and check component interfaces. |
| P4 | Diatom Frustule | Remove support coordinates, costa/areola totals, index formulas, and derived dimensions. Specify valve symmetry/separation, raphe extent, rib pitch or equally spaced positions with boundary margins, pores between ribs, and band/wall contact. Make the rib/pores counting and mirror convention unambiguous. |
| P4 | Honeycomb Lattice | Remove total cells/shapes/walls, axial walks, neighbor lists, Cartesian conversion, tilt vectors, and derived heights/extents. Specify hexagonal ring count, lattice orientation, cell dimensions/contact, inward tilt, shared-neighbor links, caps/cones, radial reinforcement, and enclosing frame/base relationships. Verify these physical relationships against tilted reference components; do not assume the current coordinate construction implements true face contact. |
| P4 | Vertebral Column | Remove direction-vector equations, positional recurrences, endpoint offsets, and evaluated totals. Supply omitted curvature inputs for each anatomical region, body/disc dimensions, initial frame, tilt convention, and local process directions. Define disc/canal placement physically and verify their geometry against the reference. |
| P4 | Cochlear Spiral | Remove parametric equations, sampling algorithm, derived total height/apex radius, and component centroid offsets. Retain an independent turn count, segmentation density, pitch, base radius, taper, sizes, handedness, and start direction. Describe partitions, entry port, modiolus, and apex through relationships. Verify the current centered modiolus against “base at zero” and apex-contact language. |
| P4 | Radiolarian Skeleton | Remove evaluated vertex/edge/shape counts and subdivision implementation instructions. Specify the geodesic refinement convention precisely through its standard topology, circumradius, node/strut/spine dimensions, equatorial ring, and a geometric icosahedron frame. Check that the reference subdivision matches that convention. |

## Implementation sequence

1. **Inventory and baseline.** Map all registered families and levels to current prompts, goldens, specs, and semantic IDs. Capture reference hashes and release-file hashes. Produce a compliance worksheet and a conflict list.
2. **Version selection and common contract.** Add strict prompt selection and isolated result paths. Establish a shared schema contract and ensure the suffix contains no construction hints. Keep legacy evaluation reproducible.
3. **Rewrite P1 and P2.** Work at family-template level so all three levels change together. Review numeric provenance and deterministic anchors against references before exporting.
4. **Rewrite P3.** Supply missing base inputs and resolve free-design versus fixed-reference inconsistencies. Check role IDs, operations, fits, and contacts.
5. **Rewrite P4.** Remove the extensive worked mathematics while retaining enough domain definitions and independent inputs to determine the intended geometry. Resolve frame/refinement/curvature ambiguities.
6. **Revise the audit.** Keep the legacy signal report reproducible. Add strict-version review records distinguishing allowed base counts, dimensions, ratios, and anchors from derived answers and worked formulas. Require a reviewed disposition for every flag; do not merely weaken regexes until they pass. Extend formula detection beyond the current regexes, which miss several explicit equations.
7. **Export and validate all 75 cases.** Generate strict cases and an audit report. Check metadata, unique identities, interpolated numeric provenance, primitive schema, deterministic regeneration, and reference agreement. Version and explain any intentional golden changes separately.
8. **Update documentation.** Document the strict prompt contract, version selection, differences from v1, unresolved/resolved reference defects, and that existing rankings apply to v1 only.
9. **Evaluate after the prompt release is finalized.** A later live evaluation must use fresh strict-version result files and record model IDs, prompt hashes, inference settings, and retry behavior. No API evaluation is needed to complete the prompt rewrite.

## Acceptance criteria

- All 25 registered families and all 75 rendered prompts receive explicit review against all five rules, including currently unflagged cases.
- Every numerical input has a documented role; no derived totals, worked coordinate formulas, or coordinate answer lists remain.
- Counts that define the task, necessary component identities, and deterministic frame anchors are retained.
- The task determines the geometry being scored; essential parameters are not hidden in generator code.
- Physical-contact and topology assertions agree with each strict golden reference. Conflicts are resolved and recorded, rather than hidden by wording.
- Legacy released evidence remains unchanged and cannot be mixed with strict-version results.
- Tests cover version isolation, complete 75-case exports, meaningful audit distinctions, deterministic references, required semantic IDs, and regression cases for known leakage/ambiguity patterns.
- Existing artifact checks and the applicable test suite pass. Current environment note: the bundled Python can run the artifact checker, but lacks pytest; provide a usable test environment during implementation.

## Deliverables

Strict prompt templates for every registered family; a versioned 75-case dataset; a per-family compliance worksheet and conflict-resolution record; updated audit and integrity checks; runner version selection with isolated output destinations; and updated benchmark documentation. Fresh model rankings are a subsequent evaluation deliverable.
