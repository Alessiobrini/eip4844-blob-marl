"""Regenerate the AIBThings paper figures at IEEE 2-column width and verify
every headline number directly from the Phase 2 results CSVs.

This is the single source of truth for the figures that ship in the paper
(the paper repository -> paper/figures/). Run end-to-end:

    python notebooks/paper_figures.py

It reads the local results/ tree (model outputs, gitignored) and writes
PDFs into paper/figures/. Headline results are reported as mean +/- s.d.
across the 5-seed runs in results/multiseed/ (final-20% steady-state
window); the phase diagram uses the 5-seed sweep in results/sweeps/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "results/sweeps/phase_diagram_20260606_150026.csv"  # 5-seed sweep
OUT = ROOT / "paper/figures"

LAMBDA = {
    "taiko": 0.803, "base": 0.361, "arbitrum_one": 0.218,
    "scroll": 0.153, "world_chain": 0.125, "metal": 0.090,
    "zircuit": 0.086, "optimism": 0.076, "starknet": 0.058,
    "paradex": 0.052, "linea": 0.051, "blast": 0.044,
    "kroma": 0.039, "zksync_era": 0.031, "river": 0.025,
    "mint": 0.015, "zora": 0.010, "mode": 0.009,
}
N5 = ["taiko", "base", "arbitrum_one", "scroll", "world_chain"]
TAIL_FRAC = 5  # final 1/TAIL_FRAC of rollouts = final 20% window

COL_W, FULL_W = 3.45, 7.0
plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,                       # match IEEEtran Computer Modern
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "figure.dpi": 300,
})

_man = pd.concat(
    [pd.read_csv(p) for p in
     sorted((ROOT / "results/multiseed").glob("manifest_*.csv"))],
    ignore_index=True,
)


def seed_dirs(config: str) -> list[Path]:
    return [Path(d) for d in _man[_man.config == config]["results_dir"]]


def _stack(dirs: list[Path], col: str) -> np.ndarray:
    """Stack a per-step column across seed runs -> array (n_seeds, n_steps)."""
    series = [pd.read_csv(d / "history.csv")[col].to_numpy() for d in dirs]
    n = min(len(s) for s in series)
    return np.vstack([s[:n] for s in series])


def _tail_band(config: str, col: str) -> tuple[float, float]:
    """Mean +/- s.d. across seeds of each run's final-20% mean."""
    vals = []
    for d in seed_dirs(config):
        df = pd.read_csv(d / "history.csv")
        vals.append(df[col].tail(max(1, len(df) // TAIL_FRAC)).mean())
    return float(np.mean(vals)), float(np.std(vals))


def fig_convergence() -> None:
    cong_dirs = seed_dirs("n5_congested")
    calib_dirs = seed_dirs("n5_calibrated")
    steps = pd.read_csv(cong_dirs[0] / "history.csv")["global_step"].to_numpy()
    n = len(steps)
    cong_b = _stack(cong_dirs, "blobs_per_block_mean")[:, :n]
    cong_lf = np.log(_stack(cong_dirs, "blob_fee_mean")[:, :n])
    cal_b = _stack(calib_dirs, "blobs_per_block_mean")[:, :n]
    cal_lf = np.log(_stack(calib_dirs, "blob_fee_mean")[:, :n])

    def band(ax, data, color, label, log=False):
        m, s = data.mean(0), data.std(0)
        if log:
            gm = np.exp(m)
            ax.semilogy(steps, gm, color=color, label=label)
            ax.fill_between(steps, np.exp(m - s), np.exp(m + s), color=color, alpha=0.25)
        else:
            ax.plot(steps, m, color=color, label=label)
            ax.fill_between(steps, m - s, m + s, color=color, alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), constrained_layout=True)
    ax = axes[0]
    band(ax, cal_b, "C0", r"Calibrated (empirical $\lambda$)")
    band(ax, cong_b, "C3", r"Congested ($\lambda{\times}100$)")
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=r"Target $b^*=3$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blobs per block")
    ax.set_title("(a) Aggregate blob supply")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    band(ax, cal_lf, "C0", r"Calibrated (empirical $\lambda$)", log=True)
    band(ax, cong_lf, "C3", r"Congested ($\lambda{\times}100$)", log=True)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blob base fee (wei)")
    ax.set_title("(b) Endogenous blob base fee")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(OUT / "fig_convergence.pdf")
    plt.close(fig)


def fig_n18() -> None:
    dirs = seed_dirs("n18")
    steps = pd.read_csv(dirs[0] / "history.csv")["global_step"].to_numpy()
    n = len(steps)
    blobs = _stack(dirs, "blobs_per_block_mean")[:, :n]
    roster = [r for r in LAMBDA if f"post_freq_{r}" in
              pd.read_csv(dirs[0] / "history.csv").columns]
    roster = sorted(roster, key=lambda r: -LAMBDA[r])
    pf = np.array([[pd.read_csv(d / "history.csv")[f"post_freq_{r}"]
                    .tail(max(1, n // TAIL_FRAC)).mean() for d in dirs]
                   for r in roster])
    pf_m, pf_s = pf.mean(1), pf.std(1)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), constrained_layout=True)
    ax = axes[0]
    m, s = blobs.mean(0), blobs.std(0)
    ax.plot(steps, m, color="C3", label=r"Aggregate supply (mean $\pm$ s.d.)")
    ax.fill_between(steps, m - s, m + s, color="C3", alpha=0.25)
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=r"Target $b^*=3$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blobs per block")
    ax.set_title(r"(a) Aggregate supply, $N{=}18$, $\lambda{\times}300$")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(roster))
    ax.bar(x, pf_m, yerr=pf_s, capsize=2, ecolor="0.3", color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "-") for r in roster], rotation=60,
                       ha="right", fontsize=6)
    ax.set_ylabel("Posting freq.")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"(b) Per-rollup posting frequency (ordered by $\lambda_i$)")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "fig_n18.pdf")
    plt.close(fig)


def fig_phase_diagram() -> None:
    df = pd.read_csv(SWEEP_CSV)
    agg = df.groupby("lambda_scale").agg(
        mean=("blobs_per_block_mean", "mean"),
        std=("blobs_per_block_mean", "std"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(COL_W, 2.7), constrained_layout=True)
    ax.errorbar(agg["lambda_scale"], agg["mean"], yerr=agg["std"],
                marker="o", color="C0", capsize=3, linewidth=1.3)
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=r"Target $b^*=3$")
    ax.axhline(6.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
               label=r"Cap $b^{\max}=6$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Arrival-rate multiplier")
    ax.set_ylabel("Steady-state blobs/block")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(OUT / "fig_phase_diagram.pdf")
    plt.close(fig)


def fig_calldata() -> None:
    dirs = seed_dirs("calldata")
    blob = np.array([[pd.read_csv(d / "history.csv")[f"blob_post_freq_{a}"]
                      .tail(max(1, len(pd.read_csv(d / "history.csv")) // TAIL_FRAC)).mean()
                      for d in dirs] for a in N5])
    cd = np.array([[pd.read_csv(d / "history.csv")[f"calldata_freq_{a}"]
                    .tail(max(1, len(pd.read_csv(d / "history.csv")) // TAIL_FRAC)).mean()
                    for d in dirs] for a in N5])
    blob_m, cd_m = blob.mean(1), cd.mean(1)
    cd_s = cd.std(1)
    wait_m = 1.0 - blob_m - cd_m

    fig, ax = plt.subplots(figsize=(COL_W, 2.8), constrained_layout=True)
    idx = np.arange(len(N5))
    ax.bar(idx, blob_m, label="Blob", color="C0", alpha=0.85)
    ax.bar(idx, cd_m, bottom=blob_m, yerr=cd_s, capsize=2, ecolor="0.3",
           label="Calldata", color="C1", alpha=0.85)
    ax.bar(idx, wait_m, bottom=blob_m + cd_m, label="Wait",
           color="lightgray", alpha=0.85)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{a.replace('_','-')}\n$\\lambda{{=}}{LAMBDA[a]:.2f}$"
                        for a in N5], fontsize=7)
    ax.set_ylabel("Action frequency")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "fig_calldata.pdf")
    plt.close(fig)


def _perroll(config: str, col_tmpl) -> dict:
    out = {}
    for a in N5:
        vals = []
        for d in seed_dirs(config):
            df = pd.read_csv(d / "history.csv")
            vals.append(df[col_tmpl(a)].tail(max(1, len(df) // TAIL_FRAC)).mean())
        out[a] = (float(np.mean(vals)), float(np.std(vals)))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_convergence()
    fig_n18()
    fig_phase_diagram()
    fig_calldata()

    print("\n===== VERIFICATION (mean +/- s.d. across 5 seeds, final-20% window) =====")
    for cfg in ["n5_congested", "n18", "calldata"]:
        m, s = _tail_band(cfg, "blobs_per_block_mean")
        gm = np.exp(np.log(_man[_man.config == cfg]["blob_fee_mean"]).mean())
        print(f"{cfg:14s} aggregate blobs/block = {m:.3f} +/- {s:.3f} | fee gmean = {gm:.2e} wei")
    print("\nN=5 congested per-rollup posting freq:")
    for a, (m, s) in _perroll("n5_congested", lambda a: f"post_freq_{a}").items():
        print(f"    {a:14s} lambda={LAMBDA[a]:.3f}  {m:.3f} +/- {s:.3f}")
    print("\nCalldata per-rollup calldata freq (NOTE: high variance, no clean size order):")
    for a, (m, s) in _perroll("calldata", lambda a: f"calldata_freq_{a}").items():
        print(f"    {a:14s} lambda={LAMBDA[a]:.3f}  {m:.3f} +/- {s:.3f}")
    df = pd.read_csv(SWEEP_CSV)
    print("\nPhase diagram (5-seed sweep):")
    for ls, g in df.groupby("lambda_scale"):
        print(f"    x{ls:<6.0f} {g['blobs_per_block_mean'].mean():.3f} +/- {g['blobs_per_block_mean'].std():.3f}")
    print("\nWrote:", *(p.name for p in sorted(OUT.glob("*.pdf"))))
