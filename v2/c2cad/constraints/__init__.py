"""Family constraint registry: build(case, ref_shapes) -> list[Constraint]."""
from .core import Ctx, Constraint, SemResult, evaluate  # noqa: F401
from . import fam_p12

BUILDERS = {
    "Spiral Staircase": fam_p12.spiral_staircase,
    "Cannonball Pyramid": fam_p12.cannonball_pyramid,
    "Voxel Grid": fam_p12.voxel_grid,
    "Domino Ring": fam_p12.domino_ring,
    "DNA Helix": fam_p12.dna_helix,
    "Flanged Pipe Joint": fam_p12.flanged_pipe_joint,
    "Suspension Bridge": fam_p12.suspension_bridge,
    "Planetary Array": fam_p12.planetary_array,
    "Cross-Braced Truss": fam_p12.cross_braced_truss,
    "Fractal Y-Tree": fam_p12.fractal_y_tree,
    "BCC Lattice": fam_p12.bcc_lattice,
    "Ball Bearing Assembly": fam_p12.ball_bearing,
}
try:
    from . import fam_p34
    BUILDERS.update(fam_p34.BUILDERS)
except ImportError:
    pass


def build(case: dict, ref) -> list[Constraint]:
    X = Ctx(case, ref)
    BUILDERS[case["family"]](case, ref, X)
    return X.out
