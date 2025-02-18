import pulp

def allocate_items_ilp(m, n, value_matrix, V):
    # 创建问题实例
    prob = pulp.LpProblem("ItemAllocation", pulp.LpMaximize)
    
    # 定义决策变量
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(m) for j in range(n)], cat='Binary')
    
    # 定义目标函数
    prob += pulp.lpSum(value_matrix[i][j] * x[(i, j)] for i in range(m) for j in range(n)), "Total_Value"
    
    # 添加约束条件
    # 每个物品只能分配给一个人
    for i in range(m):
        prob += pulp.lpSum(x[(i, j)] for j in range(n)) <= 1, f"Item_{i}_Single_Assignment"
    
    # 每个人至少得到V_j的价值
    for j in range(n):
        prob += pulp.lpSum(value_matrix[i][j] * x[(i, j)] for i in range(m)) >= V[j], f"Person_{j}_Minimum_Value"
    
    # 求解问题
    prob.solve()
    
    # 获取分配结果
    allocation = {j: [] for j in range(n)}
    total_value = pulp.value(prob.objective)
    for i in range(m):
        for j in range(n):
            if pulp.value(x[(i, j)]) == 1:
                allocation[j].append(i)
    
    return total_value, allocation

# 示例使用
if __name__ == "__main__":
    m = 6  # 物品数量
    n = 3  # 人数
    # 价值矩阵，行代表物品，列代表人
    value_matrix = [
        [12, 9, 8],  # 物品1对人的价值
        [11, 11, 6],   # 物品2
        [6, 9, 4],   # 物品3
        [3, 4, 5],    # 物品4
        [5, 7, 6],    # 物品5
        [7, 6, 8]    # 物品6
    ]
    # 每个人的最低价值需求
    V = [1, 1, 1]
    
    total_value, allocation = allocate_items_ilp(m, n, value_matrix, V)
    print(f"总价值: {total_value}")
    print("分配情况:")
    for j in range(n):
        print(f"Person_{j}: {allocation[j]}")