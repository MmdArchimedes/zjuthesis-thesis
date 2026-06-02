"""
Hand skeleton graph construction for ST-GCN.
Builds spatial-temporal graph from 26-joint hand skeleton.
Thesis reference: Direction 1 — ST-GCN/CTR-GCN gesture recognition enhancement.
"""
import numpy as np
import torch

# 26 joints (Rokid UXR enumeration):
# 0: wrist, 1: palm, 2-5: thumb (CMC/MCP/IP/Tip),
# 6-9: index (MCP/PIP/DIP/Tip), 10-13: middle,
# 14-17: ring, 18-21: pinky, 22-25: auxiliary tracking points

HAND_BONES = [
    (0, 1), (1, 2), (1, 6), (1, 10), (1, 14), (1, 18),
    (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (10, 11), (11, 12), (12, 13),
    (14, 15), (15, 16), (16, 17),
    (18, 19), (19, 20), (20, 21),
]

# Symmetric joint pairs for left/right hand consistency
SYMMETRIC_PAIRS = [
    (2, 2), (3, 3), (4, 4), (5, 5),
    (6, 18), (7, 19), (8, 20), (9, 21),
    (10, 14), (11, 15), (12, 16), (13, 17),
]

def build_hand_adjacency_matrix(strategy: str = "spatial") -> torch.Tensor:
    """
    Build adjacency matrix for 26-joint hand skeleton.

    Args:
        strategy: "spatial" (physical bones only),
                  "spatial_full" (bones + self-loops + symmetric),
                  "learnable" (identity init for dynamic learning)

    Returns:
        A: [26, 26] normalized adjacency matrix
    """
    num_joints = 26
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    for i, j in HAND_BONES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    if strategy == "spatial_full":
        # Add symmetric connections (index↔pinky, middle↔ring)
        for i, j in SYMMETRIC_PAIRS:
            if i != j:
                A[i, j] = max(A[i, j], 0.5)
                A[j, i] = max(A[j, i], 0.5)

    # Self-loops
    np.fill_diagonal(A, 1.0)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-6))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A_norm, dtype=torch.float32)


def build_ctr_adjacency(in_channels: int) -> torch.Tensor:
    """
    Build learnable adjacency for CTR-GCN style dynamic topology.
    Initialized as identity — the model learns to refine channel-wise topology.

    Args:
        in_channels: number of input channels

    Returns:
        A: [in_channels, 26, 26] channel-wise adjacency
    """
    A = torch.eye(26).unsqueeze(0).repeat(in_channels, 1, 1)
    A = A + torch.randn_like(A) * 0.01
    return A
