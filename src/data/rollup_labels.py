"""Static mapping of known rollup batcher/sequencer addresses to rollup names.

These are the Ethereum L1 addresses that post type-3 (blob) transactions
on behalf of major rollups. Verified from Etherscan labels, L2Beat, and
the OP Superchain Registry.

Note: Some rollups use multiple batcher addresses, and addresses may change
over time. This mapping covers the primary batcher addresses active during
the calibration window (March-August 2024). Update as needed.

Excluded rollups:
- Mantle: uses EigenDA, not Ethereum blobs.
- Polygon zkEVM: uses calldata, not blob transactions.
- Taiko: rotating whitelist of proposers, no single canonical batcher.
"""

from __future__ import annotations

# All addresses must be lowercase (matching loaders.py normalization).
ROLLUP_LABELS: dict[str, str] = {
    # Arbitrum One — Batch Submitter (Etherscan-labeled)
    "0xc1b634853cb333d3ad8663715b08f41a3aec47cc": "arbitrum_one",
    # Base — Batch Sender (Etherscan-labeled)
    "0x5050f69a9786f081509234f1a7f4684b5e5b76c9": "base",
    # Optimism — Batcher (Etherscan-labeled)
    "0x6887246668a3b87f54deb3b94ba47a6f63f32985": "optimism",
    # Linea — Operator (Etherscan-labeled, OpenZeppelin audit confirms blob usage)
    "0x9228624c3185fcbcf24c1c9db76d8bef5f5dad64": "linea",
    # Scroll — Batch Committer (calls ScrollChain contract with blob proofs)
    "0x054a47b9e2a22af6c0ce55020238c8fecd7d334b": "scroll",
    # Starknet — Operator (Etherscan-labeled)
    "0x2c169dfe5fbba12957bdd0ba47d9cedbfe260ca7": "starknet",
    # zkSync Era — Batcher (Etherscan-labeled, pubdata post-4844 uses blobs)
    "0x0d3250c3d5facb74ac15834096397a3ef790ec99": "zksync_era",
    # Blast — Batch Sender (Etherscan-labeled, L2Beat confirms)
    "0x415c8893d514f9bc5211d36eeda4183226b84aa7": "blast",
    # Zora — Batcher (OP Superchain Registry)
    "0x625726c858dbf78c0125436c943bf4b4be9d9033": "zora",
    # Mode — Batcher (OP Superchain Registry mode.toml)
    "0x99199a22125034c808ff20f377d91187e8050f2e": "mode",
}
"""Mapping of lowercase Ethereum address -> rollup name."""


def get_labels() -> dict[str, str]:
    """Return a copy of the rollup labels dictionary.

    Returns:
        Dict mapping lowercase address strings to rollup name strings.
    """
    return dict(ROLLUP_LABELS)
