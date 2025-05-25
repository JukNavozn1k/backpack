import copy

def backpack(weights, values, capacity):
    """
    Solves the 0/1 backpack problem using dynamic programming.
    Uses only Python standard lists and records the history of the DP table after processing each item.

    :param weights: List of item weights
    :param values:  List of item values
    :param capacity: Maximum capacity of the backpack
    :return: Tuple (max_value, selected_items, history)
    """

    # === Валидация входных данных ===
    if not isinstance(weights, list) or not isinstance(values, list):
        raise TypeError("weights and values must be lists")

    if not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")

    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")

    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    for w in weights:
        if not isinstance(w, int):
            raise TypeError("each weight must be an integer")
        if w < 0:
            raise ValueError("weights must be non-negative")

    for v in values:
        if not isinstance(v, int):
            raise TypeError("each value must be an integer")
        if v < 0:
            raise ValueError("values must be non-negative")

    # === Основная логика алгоритма ===
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    history = []

    # Record initial state (before processing any items)
    history.append(copy.deepcopy(dp))

    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
            else:
                dp[i][c] = dp[i - 1][c]
        history.append(copy.deepcopy(dp))

    max_value = dp[n][capacity]

    # Backtracking to find selected items
    selected_items = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected_items.append(i - 1)
            c -= weights[i - 1]
    selected_items.reverse()

    return max_value, selected_items, history
