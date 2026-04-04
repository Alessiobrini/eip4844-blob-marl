"""Static mapping of known rollup batcher/sequencer addresses to rollup names.

These are the Ethereum L1 addresses that post type-3 (blob) transactions
on behalf of major rollups. Verified from Etherscan labels, L2Beat,
OP Superchain Registry, and the HackMD blob adoption study.

Note: Some rollups use multiple batcher addresses, and addresses may change
over time. Taiko uses permissionless proposers, so multiple independent
addresses map to the same rollup.

Excluded:
- Mantle: uses EigenDA, not Ethereum blobs.
- Polygon zkEVM: uses calldata, not blob transactions.
"""

from __future__ import annotations

# All addresses must be lowercase (matching loaders.py normalization).
ROLLUP_LABELS: dict[str, str] = {
    # --- Arbitrum One ---
    "0xc1b634853cb333d3ad8663715b08f41a3aec47cc": "arbitrum_one",
    # --- Base ---
    "0x5050f69a9786f081509234f1a7f4684b5e5b76c9": "base",
    # --- Optimism ---
    "0x6887246668a3b87f54deb3b94ba47a6f63f32985": "optimism",
    # --- Linea (multiple submitters) ---
    "0x9228624c3185fcbcf24c1c9db76d8bef5f5dad64": "linea",  # Operator
    "0xa9268341831efa4937537bc3e9eb36dbece83c7e": "linea",  # Blob Submitter
    # --- Scroll (old + current batch committers) ---
    "0x054a47b9e2a22af6c0ce55020238c8fecd7d334b": "scroll",  # Current
    "0xcf2898225ed05be911d3709d9417e86e0b4cfc8f": "scroll",  # Old Batch Committer
    # --- Starknet ---
    "0x2c169dfe5fbba12957bdd0ba47d9cedbfe260ca7": "starknet",
    # --- zkSync Era ---
    "0x0d3250c3d5facb74ac15834096397a3ef790ec99": "zksync_era",
    # --- Blast ---
    "0x415c8893d514f9bc5211d36eeda4183226b84aa7": "blast",
    # --- Zora ---
    "0x625726c858dbf78c0125436c943bf4b4be9d9033": "zora",
    # --- Mode ---
    "0x99199a22125034c808ff20f377d91187e8050f2e": "mode",
    # --- Taiko (based rollup, permissionless proposers) ---
    "0x000000633b68f5d8d3a86593ebb815b4663bcbe0": "taiko",  # Taikobeat Proposer
    "0x41f2f55571f9e8e3ba511adc48879bd67626a2b6": "taiko",  # taiko.coinblitz.eth
    "0x66cc9a0eb519e9e1de68f6cf0aa1aa1efe3723d5": "taiko",  # Community proposer
    "0x9084ee7f65d32763ac436e4fc971e7d2cc52e672": "taiko",  # Community proposer
    "0x9a5cc6e3a3325cdc19fc76926cc9666c80139c09": "taiko",  # Community proposer/prover
    # --- World Chain ---
    "0xdbbe3d8c2d2b22a2611c5a94a9a12c2fcd49eb29": "world_chain",
    # --- Zircuit ---
    "0xaf1e4f6a47af647f87c0ec814d8032c4a4bff145": "zircuit",
    # --- Metal ---
    "0xc94c243f8fb37223f3eb2f7961f7072602a51b8b": "metal",
    # --- Paradex ---
    "0xc70ae19b5feaa5c19f576e621d2bad9771864fe2": "paradex",
    # --- Kroma ---
    "0x41b8cd6791de4d8f9e0eaf7861ac506822adce12": "kroma",
    # --- River ---
    "0x52ee324f2bcd0c5363d713eb9f62d1ee47266ac1": "river",
    # --- Mint ---
    "0x68bdfece01535090c8f3c27ec3b1ae97e83fa4aa": "mint",
}
"""Mapping of lowercase Ethereum address -> rollup name."""


def get_labels() -> dict[str, str]:
    """Return a copy of the rollup labels dictionary.

    Returns:
        Dict mapping lowercase address strings to rollup name strings.
    """
    return dict(ROLLUP_LABELS)
