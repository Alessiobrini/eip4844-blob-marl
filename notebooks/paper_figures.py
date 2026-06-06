"""Regenerate the AIBThings paper figures at IEEE 2-column width and verify
every headline number directly from the Phase 2 results CSVs.

This is the single source of truth for the figures that ship in the paper
(the paper repository -> paper/figures/). Run end-to-end:

    python notebooks/paper_figures.py

It reads the local results/ tree (model outputs, gitignored) and writes
PDFs into paper/figures/. It also prints a VERIFICATION block with the
headline statistics so the prose can be checked against the data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALIB_RUN = ROOT / "results/phase2_20260424_115247_27b811d2"      # N=5, lambda x1
CONG_RUN = ROOT / "results/phase2_20260424_142202_4d2b188e"       # N=5, lambda x100
N18_RUN = ROOT / "results/phase2_20260424_144419_93861581"        # N=18, lambda x300
CALLDATA_RUN = ROOT / "results/phase2_20260424_145217_fb53d192"   # N=5, x100, +calldata
SWEEP_CSV = ROOT / "results/sweeps/phase_diagram_20260424_144835.csv"
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

# IEEE 2-column geometry (inches): \columnwidth ~3.5, \textwidth ~7.16.
COL_W, FULL_W = 3.45, 7.0
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.linewidth": 1.2, "figure.dpi": 200,
})


def _tail_mean(df: pd.DataFrame, col: str, n: int) -> float:
    return float(df[col].tail(n).mean())


def fig_convergence() -> dict:
    calib = pd.read_csv(CALIB_RUN / "history.csv")
    cong = pd.read_csv(CONG_RUN / "history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(calib["global_step"], calib["blobs_per_block_mean"],
            label=r"Calibrated ($\lambda{\times}1$)", color="C0")
    ax.plot(cong["global_step"], cong["blobs_per_block_mean"],
            label=r"Congested ($\lambda{\times}100$)", color="C3")
    ax.axhline(3.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=r"Target $b^*=3$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blobs per block")
    ax.set_title("(a) Aggregate blob supply")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(calib["global_step"], calib["blob_fee_mean"],
                label="Calibrated", color="C0")
    ax.semilogy(cong["global_step"], cong["blob_fee_mean"],
                label=r"Congested ($\lambda{\times}100$)", color="C3")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Blob base fee (wei)")
    ax.set_title("(b) Endogenous blob base fee")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.savefig(OUT / "fig_convergence.pdf")
    plt.close(fig)
    return {
        "calib_bpb_final": _tail_mean(calib, "blobs_per_block_mean", 5),
        "cong_bpb_final": _tail_mean(cong, "blobs_per_block_mean", 5),
        "cong_fee_final": _tail_mean(cong, "blob_fee_mean", 5),
        "calib_fee_final": _tail_mean(calib, "blob_fee_mean", 5),
    }


def fig_phase_diagram() -> dict:
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
    return {row["lambda_scale"]: (row["mean"], row["std"]) for _, row in agg.iterrows()}


def fig_calldata() -> dict:
    df = pd.read_csv(CALLDATA_RUN / "history.csv")
    tail = df.tail(10)
    blob = np.array([tail[f"blob_post_freq_{a}"].mean() for a in N5])
    cd = np.array([tail[f"calldata_freq_{a}"].mean() for a in N5])
    wait = 1.0 - blob - cd

    fig, ax = plt.subplots(figsize=(COL_W, 2.8), constrained_layout=True)
    idx = np.arange(len(N5))
    ax.bar(idx, blob, label="Blob", color="C0", alpha=0.85)
    ax.bar(idx, cd, bottom=blob, label="Calldata", color="C1", alpha=0.85)
    ax.bar(idx, wait, bottom=blob + cd, label="Wait", color="lightgray", alpha=0.85)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{a.replace('_','-')}\n$\\lambda{{=}}{LAMBDA[a]:.2f}$"
                        for a in N5], fontsize=6.5)
    ax.set_ylabel("Action frequency")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "fig_calldata.pdf")
    plt.close(fig)
    return {a: (float(b), float(c)) for a, b, c in zip(N5, blob, cd)}


def verify_postfreq() -> dict:
    cong = pd.read_csv(CONG_RUN / "history.csv")
    tail = cong.tail(5)
    return {a: _tail_mean(tail, f"post_freq_{a}", 5) for a in N5}


def verify_n18() -> dict:
    df = pd.read_csv(N18_RUN / "history.csv")
    return {"bpb_final": _tail_mean(df, "blobs_per_block_mean", 5)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    conv = fig_convergence()
    pd_stats = fig_phase_diagram()
    cd_stats = fig_calldata()
    pf = verify_postfreq()
    n18 = verify_n18()

    print("\n===== VERIFICATION (headline numbers, recomputed from CSVs) =====")
    print(f"N=5 calibrated  final blobs/block : {conv['calib_bpb_final']:.3f}")
    print(f"N=5 congested   final blobs/block : {conv['cong_bpb_final']:.3f}")
    print(f"N=5 congested   final blob fee    : {conv['cong_fee_final']:.3e} wei")
    print(f"N=18 (x300)     final blobs/block : {n18['bpb_final']:.3f}")
    print("N=5 congested per-rollup posting freq (last 5 rollouts):")
    for a in N5:
        print(f"    {a:14s} lambda={LAMBDA[a]:.3f}  post_freq={pf[a]:.3f}")
    print("Phase diagram (lambda_scale -> mean blobs/block +/- sd over 2 seeds):")
    for ls, (m, s) in pd_stats.items():
        print(f"    x{ls:<6.0f} {m:.3f} +/- {s:.3f}")
    print("Calldata regime per-rollup (blob_freq, calldata_freq):")
    for a, (b, c) in cd_stats.items():
        print(f"    {a:14s} blob={b:.3f} calldata={c:.3f}")
    print("=================================================================\n")
    print("Wrote:", *(p.name for p in sorted(OUT.glob("*.pdf"))))
