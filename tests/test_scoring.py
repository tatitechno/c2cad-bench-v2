from runners.run_unified import eval_cov_geom, eval_global


def test_perfect_single_shape_scores_100():
    golden = [{"type": "sphere", "center": [0, 0, 0], "radius": 2}]
    cov, geom = eval_cov_geom(golden, golden)

    assert cov == 100
    assert geom == 100
    assert eval_global(cov, geom, 100) == 100


def test_wrong_type_gets_position_credit_only():
    golden = [{"type": "cylinder", "center": [0, 0, 0], "radius": 2, "height": 5}]
    output = [{"type": "sphere", "center": [0, 0, 0], "radius": 2}]
    cov, geom = eval_cov_geom(output, golden)

    assert cov == 100
    assert geom == 40


def test_overproduction_penalizes_coverage():
    golden = [{"type": "sphere", "center": [0, 0, 0], "radius": 2}]
    output = [
        {"type": "sphere", "center": [0, 0, 0], "radius": 2},
        {"type": "sphere", "center": [10, 0, 0], "radius": 2},
    ]
    cov, geom = eval_cov_geom(output, golden)

    assert cov == 67
    assert geom == 100


def test_degenerate_shapes_do_not_count_for_coverage():
    golden = [{"type": "sphere", "center": [0, 0, 0], "radius": 2}]
    output = [
        {"type": "sphere", "center": [0, 0, 0], "radius": 2},
        {"type": "sphere", "center": [1, 0, 0], "radius": 0},
    ]
    cov, geom = eval_cov_geom(output, golden)

    assert cov == 100
    assert geom == 100


def test_low_geometry_global_gate():
    assert eval_global(100, 0, 100) == 0
