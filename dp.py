from functools import lru_cache

def dp_item_allocation(value_matrix):
    n = len(value_matrix)    # Number of items
    k = len(value_matrix[0]) # Number of players

    @lru_cache(maxsize=None)
    def dp_value(mask, current_player):
        if mask == 0:
            return 0
        max_diff = -float('inf')
        for i in range(n):
            if mask & (1 << i):
                value = value_matrix[i][current_player]
                next_player = (current_player + 1) % k
                diff = value - dp_value(mask ^ (1 << i), next_player)
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    @lru_cache(maxsize=None)
    def dp_choice(mask, current_player):
        if mask == 0:
            return -1
        max_diff = -float('inf')
        best_choice = -1
        for i in range(n):
            if mask & (1 << i):
                value = value_matrix[i][current_player]
                next_player = (current_player + 1) % k
                diff = value - dp_value(mask ^ (1 << i), next_player)
                if diff > max_diff:
                    max_diff = diff
                    best_choice = i
        return best_choice

    # Backtracking to find the selected items
    selected = [[] for _ in range(k)]
    mask = (1 << n) - 1
    current_player = 0

    while mask != 0:
        choice = dp_choice(mask, current_player)
        if choice != -1:
            selected[current_player].append(choice)
            mask ^= (1 << choice)
            current_player = (current_player + 1) % k
        else:
            break

    return selected

# Example usage
value_matrix = [
    [12, 9, 8],  # 物品1对玩家的价值
    [11, 11, 6], # 物品2
    [6, 9, 4],   # 物品3
    [3, 4, 5],   # 物品4
    [5, 7, 6],   # 物品5
    [7, 6, 8]    # 物品6
]

selected_items = dp_item_allocation(value_matrix)
for i, items in enumerate(selected_items):
    print(f"Player {i+1} selected items: {items}")