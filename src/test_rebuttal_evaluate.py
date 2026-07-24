import numpy as np

from rebuttal_evaluate import as_velocity, evaluate


def test_flattened_prediction_layout_is_supported():
    value = np.zeros((6, 5, 4))
    assert as_velocity(value).shape == (3, 2, 5, 4)


def test_identical_fields_have_zero_consistency_error():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(2, 2, 6, 7))
    aggregate, rows = evaluate(field, field, 0.1, 0.2)
    assert aggregate["rmse"] == 0.0
    assert aggregate["e_div"] == 0.0
    assert len(rows) == 2
