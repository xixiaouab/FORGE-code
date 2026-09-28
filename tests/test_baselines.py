import numpy as np
import pytest

from forge.baselines import BM25ThresholdPolicy, dev_fallback


def test_threshold_fits_dev_and_not_test():
    scores = np.arange(6)
    utilities = np.eye(3)[[0, 0, 1, 1, 2, 2]]
    policy = BM25ThresholdPolicy.fit(scores, utilities, split="dev")
    np.testing.assert_array_equal(policy.select(scores), [0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError):
        BM25ThresholdPolicy.fit(scores, utilities, split="test")


def test_fallback_requires_positive_paired_lower_bound():
    fixed = np.tile([.2, .5, .4], (10, 1))
    assert dev_fallback(np.full(10, .6), fixed, split="dev", n_resamples=100)["use_router"]
    result = dev_fallback(np.full(10, .5), fixed, split="dev", n_resamples=100)
    assert not result["use_router"]
    assert result["fixed_action"] == 1
