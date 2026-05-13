"""Generate a small synthetic snapshot-mode dataset for ritme smoke testing.

Outputs:
- md_dummy_snapshot.tsv: metadata with columns id, host_id, time, body-site, target
- ft_dummy_snapshot.tsv: feature table with columns id, F1..F12 (relative abundances)

Designed to exercise the dynamic / temporal-snapshot path:
- ``host_col="host_id"`` -- 10 distinct hosts (>= 5 for K-fold-by-host)
- ``time_col="time"`` -- 5 ordered integer time points per host
- ``body-site`` -- a 3-level categorical so ``data_enrich_with`` exercises
  the K-fold + categorical-universe dummy alignment fix
- ``target`` -- continuous, regressable

Total rows: 50 (10 hosts x 5 time points). After snapshotting with
``n_prev=1`` and ``missing_mode="exclude"`` 40 rows survive (drop the
first time point per host).

Run from this directory:
    python generate_dummy_snapshot_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    out_dir: str | os.PathLike = Path(__file__).parent,
    n_hosts: int = 10,
    n_times: int = 5,
    n_features: int = 12,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    body_sites = np.array(["gut", "tongue", "palm"])

    rows_md = []
    rows_ft = []
    sample_idx = 0
    for h in range(n_hosts):
        # Each host gets a stable "lifestyle" body-site, drawn once.
        bs = body_sites[rng.integers(0, len(body_sites))]
        for t in range(n_times):
            sample_idx += 1
            sid = f"S{sample_idx:03d}"
            # Per-sample Dirichlet relative abundances.
            ft = rng.dirichlet(alpha=np.ones(n_features) * 0.5)
            # Target depends on a couple of current features + small noise.
            target = float(0.4 + 0.8 * ft[0] + 1.5 * ft[2] + rng.normal(0, 0.05))
            rows_md.append(
                {
                    "id": sid,
                    "host_id": f"h{h:02d}",
                    "time": int(t),
                    "body-site": bs,
                    "target": target,
                }
            )
            rows_ft.append(
                {"id": sid, **{f"F{i + 1}": ft[i] for i in range(n_features)}}
            )

    md = pd.DataFrame(rows_md).set_index("id")
    ft = pd.DataFrame(rows_ft).set_index("id")

    out = Path(out_dir)
    md.to_csv(out / "md_dummy_snapshot.tsv", sep="\t")
    ft.to_csv(out / "ft_dummy_snapshot.tsv", sep="\t")
    print(f"Wrote {out / 'md_dummy_snapshot.tsv'} ({md.shape})")
    print(f"Wrote {out / 'ft_dummy_snapshot.tsv'} ({ft.shape})")


if __name__ == "__main__":
    main()
