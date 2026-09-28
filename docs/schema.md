# JSON Primitive Schema

C2CAD-Bench uses a compact primitive schema. Each object must include a `type` and enough type-specific fields to instantiate geometry.

## Common Fields

| Field | Type | Notes |
| --- | --- | --- |
| `id` | integer or string | Optional. Scoring ignores IDs and output order. |
| `type` | string | One of the seven canonical primitive types. |
| `center` | `[x, y, z]` | Geometric centroid for centered primitives. |
| `axis` | `[x, y, z]` | Unit direction vector for axial primitives. |

## Primitive Types

| Type | Required fields |
| --- | --- |
| `box` | `center`, `size: [width, height, depth]` |
| `cylinder` | `center`, `radius`, `height`, `axis` |
| `sphere` | `center`, `radius` |
| `cone` | `center`, `base_radius`, `height`, `axis`; `top_radius` is optional and defaults to 0 where supported |
| `torus` | `center`, `ring_radius`, `tube_radius`, `axis` |
| `pipe` | `center`, `inner_radius`, `outer_radius`, `height`, `axis` |
| `beam` | `start`, `end`, `width`, `height` |

## Normalized Aliases

The scorer accepts common aliases before scoring:

- `cube`, `cuboid`, `rectangular_prism` -> `box`
- `rod`, `bar` -> `beam`
- `tube`, `hollow_cylinder` -> `pipe`
- `position`, `pos`, `origin`, `location`, `centre` -> `center`
- `dimensions`, `dims`, `width`/`height`/`depth` -> `size` for boxes
- `from`/`to`, `p1`/`p2` -> `start`/`end` for beams

Shapes with missing geometry, unrecognized types, zero dimensions, NaN, or infinite numeric fields are discarded before scoring.
