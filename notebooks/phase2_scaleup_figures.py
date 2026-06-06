"""Generate Phase 2 scale-up figures.

Outputs:
    notes/fig_phase2_n18.pdf          -- N=18 full-roster convergence
    notes/fig_phase2_phase_diagram.pdf -- steady-state b/b vs lambda_scale
    notes/fig_phase2_calldata.pdf      -- blob vs calldata usage per rollup
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N18_RUN = ROOT / "results/phase2_20260424_144419_93861581"
CALLDATA_RUN = ROOT / "results/phase2_20260424_145217_fb53d192"
SWEEP_CSV = ROOT / "results/sweeps/phase_diagram_20260424_144835.csv"
OUT = ROOT / "notes"

LAMBDA = {
    "taiko": 0.803, "base": 0.361, "arbitrum_one": 0.218,
    "scroll": 0.153, "world_chain": 0.125, "metal": 0.090,
    "zircuit": 0.086, "optimism": 0.076, "starknet": 0.058,
    "paradex": 0.052, "linea": 0.051, "blast": 0.044,
    "kroma": 0.039, "zksync_era": 0.031, "river": 0.025,
    "mint": 0.015, "zora": 0.010, "mode": 0.009,
}


def fig_n18_convergence() -> None:
    df = pd.read_csv(N18_RUN / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(df["global_step"], df["blobs_per_block_mean"], color="C3")
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label="EIP-4844 target $b^*=3$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blobs per block")
    ax.set_title(r"(a) Aggregate supply, $N{=}18$, $\lambda{\times}300$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels_in_df = [l for l in LAMBDA if f"post_freq_{l}" in df.columns]
    tail = df.tail(5)
    pf = np.array([tail[f"post_freq_{l}"].mean() for l in labels_in_df])
    lam = np.array([LAMBDA[l] for l in labels_in_df])
    order = np.argsort(-lam)
    ax.bar(range(len(order)), pf[order], color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels_in_df[i].replace("_", "-") for i in order],
                       rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("Posting freq.\\ (last 10\\% of training)")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"(b) Per-rollup posting frequency (ordered by $\lambda_i$)")
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(OUT / "fig_phase2_n18.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_phase2_n18.pdf'}")


def fig_phase_diagram() -> None:
    df = pd.read_csv(SWEEP_CSV)
    agg = df.groupby("lambda_scale").agg(
        mean=("blobs_per_block_mean", "mean"),
        std=("blobs_per_block_mean", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    ax.errorbar(agg["lambda_scale"], agg["mean"], yerr=agg["std"],
                marker="o", color="C0", capsize=3, linewidth=1.5)
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label="EIP-4844 target $b^*=3$")
    ax.axhline(6.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
               label="Cap $b^{\\max}=6$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Arrival-rate multiplier $\lambda$-scale")
    ax.set_ylabel("Steady-state blobs per block")
    ax.set_title(
        r"Phase diagram, $N{=}5$, mean$\pm$sd over 2 seeds, 30k training steps"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.savefig(OUT / "fig_phase2_phase_diagram.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_phase2_phase_diagram.pdf'}")


def fig_calldata_usage() -> None:
    df = pd.read_csv(CALLDATA_RUN / "history.csv")
    tail = df.tail(10)
    agents = ["taiko", "base", "arbitrum_one", "scroll", "world_chain"]

    blob_freq = np.array([tail[f"blob_post_freq_{a}"].mean() for a in agents])
    cd_freq = np.array([tail[f"calldata_freq_{a}"].mean() for a in agents])
    wait_freq = 1.0 - blob_freq - cd_freq
    lambdas = np.array([LAMBDA[a] for a in agents])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    idx = np.arange(len(agents))
    ax.bar(idx, blob_freq, label="Blob post", color="C0", alpha=0.85)
    ax.bar(idx, cd_freq, bottom=blob_freq, label="Calldata post",
           color="C1", alpha=0.85)
    ax.bar(idx, wait_freq, bottom=blob_freq + cd_freq,
           label="Wait", color="lightgray", alpha=0.85)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{a.replace('_','-')}\n$\\lambda{{=}}{l:.2f}$"
         for a, l in zip(agents, lambdas)],
        fontsize=8,
    )
    ax.set_ylabel("Action frequency (last 10\\% of training)")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Blob vs. calldata choice, $100\times$ arrival-rate regime")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(OUT / "fig_phase2_calldata.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_phase2_calldata.pdf'}")


if __name__ == "__main__":
    fig_n18_convergence()
    fig_phase_diagram()
    fig_calldata_usage()
