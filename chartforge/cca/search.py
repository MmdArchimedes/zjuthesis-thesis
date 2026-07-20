"""
CCA Combinatorial Search — search the algebraic space for optimal chart compositions.

Given a user intent (CIF), this module searches the space of possible chart
compositions using CCA operations to find the best multi-chart layout.
"""

from typing import List, Optional, Dict
from ..pcg.sampler import ChartSpec
from ..cif.schema import CIFTriple
from ..svas.scorer import svas
from .algebra import compose, layer, transform, change_color_scheme


def search_compositions(
    base_charts: List[ChartSpec],
    cif: CIFTriple,
    max_depth: int = 3,
    beam_size: int = 5,
) -> List[ChartSpec]:
    """Search the CCA algebraic space for optimal chart compositions.

    Uses beam search over composition operations (compose, layer, transform)
    to find the best multi-chart layout for a given intent.

    Args:
        base_charts: Base chart candidates from PCG beam search
        cif: User intent (CIF triple)
        max_depth: Maximum composition depth
        beam_size: Beam size for search

    Returns:
        Ranked list of composite chart candidates
    """
    # Initialize beam with base charts
    beam = [(c, svas(c, cif)) for c in base_charts]
    beam.sort(key=lambda x: -x[1])
    beam = beam[:beam_size]

    for depth in range(max_depth):
        candidates = list(beam)

        for i, (c1, _) in enumerate(beam):
            for j, (c2, _) in enumerate(beam):
                if i == j:
                    continue

                # Try layering
                try:
                    layered = layer(c1, c2)
                    score = svas(layered, cif)
                    candidates.append((layered, score))
                except AssertionError:
                    pass

                # Try composition
                try:
                    composed = compose(c1, c2)
                    score = svas(composed, cif)
                    candidates.append((composed, score))
                except Exception:
                    pass

                # Try color transformations
                for scheme in ["tableau10", "viridis", "blues", "reds"]:
                    transformed = transform(c1, lambda c, s=scheme: change_color_scheme(c, s))
                    score = svas(transformed, cif)
                    candidates.append((transformed, score))

        # Keep top-k
        candidates.sort(key=lambda x: -x[1])
        beam = candidates[:beam_size]

        # Stop if no improvement
        if len(beam) > 0 and beam[0][1] <= beam[0][1] * 1.01:
            break

    return [c for c, _ in beam]


def find_optimal_layout(
    chart_specs: List[ChartSpec],
    num_charts: int,
) -> List[List[float]]:
    """Determine optimal AR spatial layout for multiple charts.

    Returns list of [x, y, z] positions for each chart in AR space.
    """
    positions = []
    spacing = 1.0  # meters between charts

    if num_charts == 1:
        positions.append([0.0, 0.0, 1.5])
    elif num_charts <= 3:
        # Horizontal row
        start_x = -(num_charts - 1) * spacing / 2
        for i in range(num_charts):
            positions.append([start_x + i * spacing, 0.0, 1.5])
    elif num_charts <= 6:
        # 2-row grid
        per_row = (num_charts + 1) // 2
        for row in range(2):
            n_in_row = min(per_row, num_charts - row * per_row)
            start_x = -(n_in_row - 1) * spacing / 2
            for col in range(n_in_row):
                positions.append([
                    start_x + col * spacing,
                    -row * 0.7,  # stagger vertically
                    1.5,
                ])
    else:
        # 3×3 grid
        for i in range(num_charts):
            row = i // 3
            col = i % 3
            positions.append([
                (col - 1) * spacing,
                (1 - row) * 0.7,
                1.5,
            ])

    return positions
