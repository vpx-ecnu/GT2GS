import networkx as nx

def allocate_items(m, n, value_matrix):
    # 创建有向图
    G = nx.DiGraph()
    
    items = [f'item_{i}' for i in range(1, m+1)]
    people = [f'person_{j}' for j in range(1, n+1)]
    
    # 添加源点S和汇点T
    G.add_node('S')
    G.add_node('T')
    
    # 添加物品节点，并从S到每个物品添加容量为1、权重为0的边
    for item in items:
        G.add_edge('S', item, capacity=1, weight=0)
    
    # 添加人节点，并设置每个人的需求为1
    for person in people:
        G.add_node(person)
        G.add_edge(person, 'T', capacity=n, weight=0)  # 从人到T的边
        G.nodes[person]['demand'] = 1  # 每个人需求1个物品
    
    # 添加物品到人的边，权重为负的价值
    for i, item in enumerate(items):
        for j, person in enumerate(people):
            value = value_matrix[i][j]
            G.add_edge(item, person, capacity=1, weight=-value)  # 负价值用于最小成本流
    
    # 设置源点和汇点的需求
    G.nodes['S']['demand'] = -m  # 供给m个物品
    G.nodes['T']['demand'] = m - n  # 汇点接收剩余的流量
    
    # 计算最小成本流
    flow_cost = nx.min_cost_flow_cost(G)
    
    # 获取流分配
    flow_dict = nx.min_cost_flow(G)
    
    # 计算总价值并记录分配情况
    total_value = 0
    allocation = {person: [] for person in people}
    for item in items:
        for person in people:
            if flow_dict[item].get(person, 0) == 1:
                value = value_matrix[items.index(item)][people.index(person)]
                total_value += value
                allocation[person].append(item)
    
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
    
    total_value, allocation = allocate_items(m, n, value_matrix)
    print(f"总价值: {total_value}")
    print("分配情况:")
    for person, items in allocation.items():
        print(f"{person}: {items}")