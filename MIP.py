import gurobipy as gp
from gurobipy import GRB

# 数据定义
m = 5  # 物品数量
k = 3  # 人数
value_matrix = [
    [12, 11, 6, 3, 5],  # 玩家0对每个物品的价值评估
    [9, 11, 9, 4, 7],   # 玩家1对每个物品的价值评估
    [8, 6, 4, 5, 6]     # 玩家2对每个物品的价值评估
]
epsilon = 1e-6  # 小的正数，确保严格大于

# 创建模型
model = gp.Model('Allocation')

# 定义变量
x = model.addVars(k, m, vtype=GRB.BINARY, name='x')  # 玩家i选择物品j
V = model.addVars(k, lb=0, name='V')  # 玩家i获得的总价值
V_max = model.addVars(k, lb=0, name='V_max')  # 除了玩家i之外的最大价值
gamma = model.addVar(lb=0, name='gamma')  # 最小差值
y = model.addVars(k, k, vtype=GRB.BINARY, name='y')  # 辅助变量选择V_max

# 约束条件

# 每个物品只能被一个人选择
for j in range(m):
    model.addConstr(sum(x[i, j] for i in range(k)) == 1)

# 每个人获得的总价值
for i in range(k):
    model.addConstr(V[i] == sum(value_matrix[i][j] * x[i, j] for j in range(m)))

# V_max[i] >= V[j] for all j != i
for i in range(k):
    for j in range(k):
        if j != i:
            model.addConstr(V_max[i] >= V[j])

# V_max[i] <= V[j] + M * (1 - y[i,j]) for all j != i
max_sum_v = max(sum(value_matrix[i]) for i in range(k))
M = max_sum_v + 1
for i in range(k):
    for j in range(k):
        if j != i:
            model.addConstr(V_max[i] <= V[j] + M * (1 - y[i, j]))

# sum y[i,j] for j != i == 1
for i in range(k):
    model.addConstr(sum(y[i, j] for j in range(k) if j != i) == 1)

# gamma <= V[i] - V_max[i] for all i
for i in range(k):
    model.addConstr(gamma <= V[i] - V_max[i])

# gamma >= epsilon
model.addConstr(gamma >= epsilon)

# 设置目标函数
model.setObjective(gamma, GRB.MAXIMIZE)

# 求解模型
model.optimize()

# 检查模型状态
print(f'Model status: {model.Status}')

# 输出结果
if model.Status == GRB.OPTIMAL:
    for i in range(k):
        print(f'Player {i} takes items: ', [j for j in range(m) if x[i, j].getAttr('X') >= 0.99])
        print(f'Player {i} total value: {V[i].getAttr("X")}')
    print(f'Minimum difference: {gamma.getAttr("X")}')
elif model.Status == GRB.INFEASIBLE:
    # 尝试移除 gamma >= epsilon 约束，寻找次优解
    model.remove(model.getConstrByName('gamma_lower_bound'))
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        for i in range(k):
            print(f'Player {i} takes items: ', [j for j in range(m) if x[i, j].getAttr('X') >= 0.99])
            print(f'Player {i} total value: {V[i].getAttr("X")}')
        print(f'Minimum difference: {gamma.getAttr("X")}')
    else:
        print('No feasible solution found even after relaxing constraints.')
else:
    print('Model is not optimal. Status code:', model.Status)