"""Generate Phase 2 report figures from two training runs.

Outputs:
    notes/fig_phase2_convergence.pdf   -- aggregate blobs/block and blob fee
    notes/fig_phase2_postfreq.pdf       -- per-agent posting frequency vs lambda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALIB_RUN = ROOT / "results/phase2_20260424_115247_27b811d2"
CONG_RUN = ROOT / "results/phase2_20260424_142202_4d2b188e"
OUT = ROOT / "notes"

LAMBDA = {
    "taiko": 0.803,
    "base": 0.361,
    "arbitrum_one": 0.218,
    "scroll": 0.153,
    "world_chain": 0.125,
}

AGENT_ORDER = ["taiko", "base", "arbitrum_one", "scroll", "world_chain"]


def _load(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "history.csv")


def fig_convergence() -> None:
    calib = _load(CALIB_RUN)
    cong = _load(CONG_RUN)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(calib["global_step"], calib["blobs_per_block_mean"],
            label="Calibrated (baseline $\\lambda$)", color="C0")
    ax.plot(cong["global_step"], cong["blobs_per_block_mean"],
            label=r"Congested ($100\times\ \lambda$)", color="C3")
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label="EIP-4844 target $b^*=3$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blobs per block (rollout mean)")
    ax.set_title("(a) Aggregate blob supply")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(calib["global_step"], calib["blob_fee_mean"],
                label="Calibrated", color="C0")
    ax.semilogy(cong["global_step"], cong["blob_fee_mean"],
                label=r"Congested ($100\times\ \lambda$)", color="C3")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blob base fee (wei, log scale)")
    ax.set_title("(b) Endogenous blob base fee")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.savefig(OUT / "fig_phase2_convergence.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_phase2_convergence.pdf'}")


def fig_postfreq() -> None:
    cong = _load(CONG_RUN)
    tail = cong.tail(5)
    pf = {lbl: float(tail[f"post_freq_{lbl}"].mean()) for lbl in AGENT_ORDER}

    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    lambdas = [LAMBDA[l] for l in AGENT_ORDER]
    freqs = [pf[l] for l in AGENT_ORDER]

    colors = ["C0", "C1", "C2", "C3", "C4"]
    ax.scatter(lambdas, freqs, s=80, c=colors, zorder=3)
    for l, lam, f, c in zip(AGENT_ORDER, lambdas, freqs, colors):
        ax.annotate(l.replace("_", "-"), xy=(lam, f),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel(r"Empirical arrival rate $\lambda_i$ (tx/block)")
    ax.set_ylabel("Learned posting frequency (last 10\\% of training)")
    ax.set_title("Posting frequency ordered by arrival rate (congested regime)")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, which="both")

    fig.savefig(OUT / "fig_phase2_postfreq.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_phase2_postfreq.pdf'}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_convergence()
    fig_postfreq()
