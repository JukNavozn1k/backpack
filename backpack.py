import copy

def backpack(weights, values, capacity):
    """
    Solves the 0/1 backpack problem using dynamic programming.
    Uses only Python standard lists and records the history of the DP table after processing each item.
    
    :param weights: List of item weights
    :param values:  List of item values
    :param capacity: Maximum capacity of the backpack
    :return: Tuple (max_value, selected_items, history)
             where max_value is the maximum achievable value,
             selected_items is the list of chosen item indices,
             history is a list of DP tables after each item is processed.
    """
    n = len(weights)
    # Initialize DP table: (n+1) x (capacity+1)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    history = []
    
    # Record initial state (before processing any items)
    history.append(copy.deepcopy(dp))
    
    # Fill DP table
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i-1][c], dp[i-1][c-w] + v)
            else:
                dp[i][c] = dp[i-1][c]
        # Record state after processing item i
        history.append(copy.deepcopy(dp))
    
    # Maximum value is in dp[n][capacity]
    max_value = dp[n][capacity]
    
    # Reconstruct which items were taken
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            selected.append(i - 1)
            c -= weights[i-1]
    selected.reverse()
    
    return max_value, selected, history

# Пример использования
if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values  = [3, 4, 5, 6]
    capacity = 5

    max_val, chosen_items, dp_history = backpack(weights, values, capacity)

    print(f"Max value: {max_val}")               # 7
    print(f"Chosen item indices: {chosen_items}") # [0, 1]
    print(f"Snapshots stored: {len(dp_history)}")  # 5 (0…4)
    # Для просмотра DP-таблицы после всех итераций:
    for row in dp_history[-1]:
        print(row)
