from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(df: pd.DataFrame, test: float, val: float, hmm_frac_of_train: float,
                seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    rate = df.groupby("instance_id")["target"].mean()
    strata = pd.cut(rate, bins=[-0.01, 0.0, 0.25, 0.5, 0.75, 1.0], labels=False)

    assignment: dict[str, str] = {}
    for _, ids in rate.groupby(strata).groups.items():
        ids = np.array(list(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = int(round(n * test)), int(round(n * val))
        n_hmm = int(round((n - n_test - n_val) * hmm_frac_of_train))
        cuts = np.cumsum([n_test, n_val, n_hmm])
        for name, part in zip(["test", "val", "hmm_train", "clf_train"], np.split(ids, cuts)):
            assignment.update({i: name for i in part})
    return df["instance_id"].map(assignment).rename("split")
