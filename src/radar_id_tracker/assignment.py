"""Deterministic rectangular minimum-cost assignment with unmatched rows."""

from __future__ import annotations

from math import isfinite


def assign(
    costs: list[list[float | None]],
    detection_count: int,
    unmatched_cost: float = 1.01,
) -> list[tuple[int, int]]:
    """Return matched (track-row, detection-column) pairs.

    ``None`` forbids an edge. Every track receives its own dummy unmatched
    column, so the solver never has to force a bad detection match.
    """
    row_count = len(costs)
    if row_count == 0:
        return []
    if detection_count < 0 or not isfinite(unmatched_cost):
        raise ValueError("invalid assignment dimensions or unmatched cost")
    if any(len(row) != detection_count for row in costs):
        raise ValueError("each cost row must have detection_count columns")

    forbidden = 1e9
    column_count = detection_count + row_count
    matrix = []
    for row_index, row in enumerate(costs):
        values = [
            forbidden if value is None else float(value)
            for value in row
        ]
        if any(not isfinite(value) for value in values):
            raise ValueError("costs must be finite or None")
        values.extend(
            unmatched_cost if dummy_index == row_index else forbidden
            for dummy_index in range(row_count)
        )
        matrix.append(values)

    # Hungarian shortest augmenting path formulation; 1-based indices.
    potentials_rows = [0.0] * (row_count + 1)
    potentials_cols = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_col = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        column = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    matrix[current_row - 1][candidate - 1]
                    - potentials_rows[current_row]
                    - potentials_cols[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_col[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    potentials_rows[matched_row[candidate]] += delta
                    potentials_cols[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = previous_col[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    pairs = []
    for column in range(1, detection_count + 1):
        row = matched_row[column] - 1
        if row >= 0 and matrix[row][column - 1] < unmatched_cost:
            pairs.append((row, column - 1))
    pairs.sort()
    return pairs
