import numpy as np

from spiderstate.utils import find_pivots_in_matrix, well_ordered_ft_cat_state_data


def flag_by_construction(H: np.ndarray, d: int):
    N = H.shape[1]
    t = (d - 1) // 2

    pivots, rows_without_pivots = find_pivots_in_matrix(H)
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(H, axis=1)
    x_spiders = np.sum(H[:, non_pivots], axis=0) + 1

    z_data = [well_ordered_ft_cat_state_data(zs, t) for zs in z_spiders]
    x_data = [well_ordered_ft_cat_state_data(xs, t) for xs in x_spiders]

