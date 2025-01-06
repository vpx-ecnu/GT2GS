def main():
    import sys
    sys.setrecursionlimit(1000000)
    
    m = 5  # Number of items
    k = 3  # Number of players
    value_matrix = [
        [12, 9, 8],   # 物品0
        [11, 11, 6],  # 物品1
        [6, 9, 4],    # 物品2
        [3, 4, 5],    # 物品3
        [5, 7, 6],    # 物品4
        # [7, 6, 8]     # 物品5
    ]
    
    # 转置价值矩阵，使得 v[p][j] = value_matrix[j][p]
    v = [[value_matrix[j][p] for j in range(m)] for p in range(k)]
    
    memo = {}
    choices = {}
    
    def dp(s, p):
        if s == 0:
            return 0
        # 记忆化
        if (s, p) in memo:
            return memo[(s, p)]
        
        max_diff = -float('inf')
        best_j = -1
        # 枚举所有可选物品j
        # s & (1 << j) s状态的j是否可选
        for j in range(m):
            if s & (1 << j):
                # 当前玩家选择物品j，下一个玩家是 (p+1)%k
                diff = v[p][j] - dp(s ^ (1 << j), (p + 1) % k)
                if diff > max_diff:
                    max_diff = diff
                    best_j = j
        memo[(s, p)] = max_diff
        choices[(s, p)] = best_j
        return max_diff
    
    dp((1 << m) - 1, 0)
    
    def build_choices(s, p):
        if s == 0:
            return []
        j = choices[(s, p)]
        next_s = s ^ (1 << j)
        next_p = (p + 1) % k
        return [j] + build_choices(next_s, next_p)
    
    choice_order = build_choices((1 << m) - 1, 0)
    players = [[] for _ in range(k)]
    for i, j in enumerate(choice_order):
        players[i % k].append(j)
    
    # 计算每个玩家的总价值
    total_values = [0 for _ in range(k)]
    for i in range(k):
        for j in players[i]:
            total_values[i] += v[i][j]
    
    # 计算净价值差
    net_diff = total_values[0] - total_values[1] - total_values[2]
    
    print("选择顺序:", choice_order)
    print("玩家分配的物品:", players)
    print("玩家的总价值:", total_values)
    print("净价值差:", net_diff)

if __name__ == "__main__":
    main()