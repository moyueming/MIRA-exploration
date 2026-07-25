import numpy as np

from plot_galaxy_extrinsic_reward_template import (
    aggregate_seed_curves,
    errorbar_indices,
)


def test_aggregate_smooths_each_seed_before_cross_seed_statistics():
    seeds = np.array(
        [
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 6.0, 8.0, 10.0],
        ]
    )

    mean, sd = aggregate_seed_curves(seeds, window=2)

    np.testing.assert_allclose(mean, [2.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(sd, [2.0, 2.0, 2.0, 2.0])


def test_errorbar_indices_include_first_regular_checkpoints_and_last():
    episodes = np.arange(1, 1001)

    indices = errorbar_indices(episodes, every=50)
    selected = episodes[indices]

    assert selected[0] == 1
    assert selected[-1] == 1000
    assert set(range(50, 1001, 50)).issubset(set(selected))
    assert len(selected) == 21
