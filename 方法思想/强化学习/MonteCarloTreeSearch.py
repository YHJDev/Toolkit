import random
import math

# 定义每个搜索组件的候选项
perturbation_intensity_candidate = [0.5, 0.1, 0.05, 0.01]
perturbation_probability_candidate = [0.1, 0.3, 0.7]
distance_aggregator_candidate = ["5", "10", "30"]
feature_aggregator_candidate = ["GCNConv", "SGConv", "None"]
updator_dimension_candidate = [128, 256, 512, 1024]
updator_activation_candidate = ["LeakyRelu", "Relu", "Relu6"]
fusion_candidate = ["cat", "sum", "mean", "max", "weighted_cat", "weighted_sum"]

# 将所有组件的候选项按照指定顺序组合成列表
components_candidates = [
    perturbation_intensity_candidate,
    perturbation_probability_candidate,
    distance_aggregator_candidate,
    feature_aggregator_candidate,
    updator_dimension_candidate,
    updator_activation_candidate,
    fusion_candidate,
]

# 假设的评估函数，需要替换为实际的评估函数
def estimator(architecture):
    # 在实际应用中，此函数应返回给定架构的验证评估值 val_score
    # 这里为了演示，返回一个随机值
    return random.uniform(0, 1)

# 定义节点类，表示蒙特卡洛树搜索中的一个节点
class Node:
    def __init__(self, parent=None, component_index=0, component_value=None):
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.component_index = component_index  # 当前节点对应的组件索引
        self.component_value = component_value  # 当前节点选择的组件值
        self.visits = 0  # 节点被访问的次数
        self.total_value = 0.0  # 节点的累计评估值
        # 未尝试的候选项列表（深拷贝避免原列表被修改）
        if component_index < len(components_candidates):
            self.untried_values = components_candidates[component_index][:]
        else:
            self.untried_values = []
        self._is_fully_expanded = False  # 节点是否已完全扩展，命名加下划线

    # 判断节点是否是终端节点（所有组件都已选择）
    def is_terminal(self):
        return self.component_index == len(components_candidates)

    # 判断节点是否已完全扩展，方法名与属性名区分
    def is_fully_expanded(self):
        return self._is_fully_expanded

    # 使用 UCB1 算法选择最佳子节点
    def selection(self, exploration_weight=math.sqrt(2)):
        best_score = float('-inf')
        best_children = []
        for child in self.children:

            if child.visits == 0:
                # 如果子节点未被访问，赋予无限大的 UCB1 值
                score = float('inf')
            else:
                # 确保父节点的访问次数不为零
                parent_visits = max(self.visits, 1)
                # 计算平均价值
                exploit = child.total_value / child.visits
                # 计算探索项
                explore = exploration_weight * math.sqrt(math.log(parent_visits) / child.visits)
                # 计算 UCB1 值
                score = exploit + explore
            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_children = [child]
                best_score = score
        # 随机选择得分最高的子节点

        return random.choice(best_children)

    # 扩展节点，创建一个新的子节点
    def expansion(self):
        # 从未尝试的候选项中选择一个值
        value = self.untried_values.pop()
        # 创建新的子节点
        child_node = Node(
            parent=self,
            component_index=self.component_index + 1,
            component_value=value
        )
        self.children.append(child_node)
        # 如果没有未尝试的值，标记为已完全扩展
        if not self.untried_values:
            self._is_fully_expanded = True
        return child_node

# 树策略：选择和扩展节点
def monte_carlo_tree_construction(node):
    while not node.is_terminal():
        if not node.is_fully_expanded():
            # 节点未完全扩展，进行扩展
            return node.expansion()
        else:
            # 节点已完全扩展，选择最佳子节点
            node = node.selection()
    return node

# 默认策略：模拟（随机选择剩余的组件）
def simulation(node):
    current_node = node
    architecture = []
    # 回溯获取已选择的组件值
    while current_node.parent is not None:
        architecture.append(current_node.component_value)
        current_node = current_node.parent
    architecture = architecture[::-1]  # 反转列表，得到正确的顺序

    # 随机选择剩余的组件，构建完整的架构
    for index in range(node.component_index, len(components_candidates)):
        value = random.choice(components_candidates[index])
        architecture.append(value)
    # 使用评估函数获取 val_score
    val_score = estimator(architecture)
    return val_score

# 回溯更新节点的访问次数和累计评估值，增加折扣因子
def backpropagation(node, reward, discount_factor=0.9):
    """
    Args:
        node: 当前节点
        reward: 模拟得到的评估值
        discount_factor: 折扣因子，取值范围 [0, 1]
    """
    current_reward = reward  # 当前回报
    while node is not None:
        node.visits += 1
        node.total_value += current_reward
        # 折扣回报
        current_reward *= discount_factor
        node = node.parent

# 蒙特卡洛树搜索算法
def mcts(root, iterations, discount_factor=0.9):
    for _ in range(iterations):
        # 选择和扩展
        leaf = monte_carlo_tree_construction(root)
        # 模拟
        reward = simulation(leaf)
        # 回溯更新，加入折扣因子
        backpropagation(leaf, reward, discount_factor)

    # 从根节点开始，构建最佳架构
    best_architecture = []
    node = root
    while not node.is_terminal():
        # 在每一层，选择访问次数最多的子节点
        if node.children:
            node = max(node.children, key=lambda c: c.visits)
            best_architecture.append(node.component_value)
        else:
            # 如果没有子节点，跳出循环，避免错误
            break
    # 获取最佳架构的平均评估值
    if node.visits > 0:
        best_val_score = node.total_value / node.visits
    else:
        best_val_score = 0
    return best_architecture, best_val_score

# 设置迭代次数
max_iterations = 1000  # 可根据实际需求调整

# 初始化根节点
root_node = Node()

# 执行蒙特卡洛树搜索，设置折扣因子
best_architecture, best_val_score = mcts(root_node, max_iterations, discount_factor=0.9)

# 输出最佳架构和验证评估值
print("最佳 val_score:", best_val_score)
print("最佳 GNN 架构:", best_architecture)
