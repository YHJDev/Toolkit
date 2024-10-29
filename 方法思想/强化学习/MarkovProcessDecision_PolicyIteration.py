import numpy as np

# 问题定义:
# 在一个3*3的空间里,一个小球在最左上角,需要找到小球的最优行动策略到空间里的最右下角,小球每次移动一格,小球有上下左右移动的动作
# 定义状态空间为:小球处于的空间位置
# 定义动作空间为:小球可以上下左右移动
# 定义状态转移概率为: 当执行动作是向上移动时,100%转移至状态next_state = (max(0, x-1), y):物理含义如果没有碰到墙，向上移动一格
# 当执行动作是向下向左向右同上
# 定义小球的即时奖励函数为:小球达到指定地点+10,小球碰墙-1,小球没有碰墙0.
# 定义折扣因子0.9

# step1: 构建状态空间
states = [(i, j) for i in range(3) for j in range(3)]

# step2: 初始化状态价值函数
# 使用0初始化转态空间中每个状态的价值函数值
initialization_state_value_function = {state: 0 for state in states}

# step3: 构建动作空间
actions = ['up', 'down', 'left', 'right']

# 根据转态空间与动作空间使用随机初始化方法得到一个确定性策略,即该策略给每一个状态一个确定的动作pi(a|s)=1，
# 本问题的策略是一个字典组成,键表示小球所处的状态(位置),值表示小球在此状态下最优的动作
# step4: 构建策略——>定义状态与该状态可执行动作关系
policy = {state: np.random.choice(actions) for state in states}

# step5: 构建折扣因子
gamma = 0.9

# 定义最大迭代次数
n_iterations = 100

# step6: 基于先验知识构建奖励函数,R(s,a)——>r
def get_reward(state, action, next_state):
    if next_state == (2, 2):
        return 10
    elif state == next_state:  # 碰墙
        return -1
    else:
        return 0

# step7:根据先验知识，已形式化的状态空间S，已形式化的动作空间A，构建状态转移函数,P(s'|s,a)——>(s',p)
def state_transition_function(state, action):

    x, y = state
    if action == 'up':
        transition_probability = 1
        next_state = (max(0, x-1), y)

    elif action == 'down':
        transition_probability = 1
        next_state = (min(2, x+1), y)

    elif action == 'left':
        transition_probability = 1
        next_state = (x, max(0, y-1))

    elif action == 'right':
        transition_probability = 1
        next_state = (x, min(2, y+1))

    return next_state, transition_probability


# step8: 构建"策略迭代方法"优化MDP求解最优策略
# step8.1: 使用贝尔曼期望方程构建基于当前策略更新状态价值函数过程
def update_state_value_function_based_on_policy(policy, gamma, n, state_value_function):

    state_value_function_new = {}

    # 策略评估必须准确:策略评估步骤的准确性至关重要。在策略迭代中，通常需要执行多次迭代，直到价值函数收敛或达到可接受的精度水平。
    # n 为此次策略评估迭代的次数
    for _ in range(n): # 充分的策略评估, 执行多次迭代（例如 100 次）可以使价值函数收敛或接近收敛。这提供了一个更准确的v_pi(s) 估计，使得策略改进步骤能够找到最优策略。

    # for _ in range(1): 策略评估不足情况，仅执行一次迭代不足以使价值函数准确地反映遵循当前策略的长期回报。结果是，随后的策略改进步骤基于v_pi(s) 的一个较差估计，从而导致次优的策略

        # 递归更新整个状态空间中每一个状态的价值函数值
        for state in states:

            # 根据策略函数获取该状态的动作
            action = policy[state]
            # 通过先验的状态转移概率函数基于现有状态state和执行的动作action获得下一个状态next_state与状态转移概率
            next_state, transition_probability = state_transition_function(state, action)
            # 根据当前状态state与执行的动作action输入奖励函数获得此刻的即时奖励，本问题中需要用next_state计算即时奖励。
            reward = get_reward(state, action, next_state)
            # 根据贝尔曼期望方程更新此状态函数每一个状态价值,贝尔曼期望方程构建了状态价值函数与动作价值函数的递归关系，使得可以通过迭代方式基于当前策略更新每个状态价值与动作价值
            # 在此问题中,状态价值函数中没有动作求和号sigma，没有策略概率函数pi(a|s),因为该策略是一个确定性策略.
            state_value_function_new[state] = reward + gamma * transition_probability * state_value_function[next_state]

        state_value_function = state_value_function_new.copy()
    # 经过n次迭代后，state_value_function能较为准确地反应当前policy为每个状态带来的长期价值(期望回报)
    return state_value_function

# step8.2:为每个一个状态的所有可能执行动作基于已更新的状态价值函数计算动作价值函数值,并基于最大的动作价值为每一个状态更新最优动作,
# 更新状态执行动作的过程就是策略改进过程
def policy_improvement(new_value_function_based_on_policy, gamma):

    new_policy = policy.copy()
    # 对每个状态state可能执行的动作action进行考察
    for state in states:

        best_action = None
        best_action_value = float('-inf')

        # 基于以下信息重新计算每个状态下每个动作的动作价值函数值
        # 1.状态state;
        # 2.此状态可执行的动作action;
        # 3.奖励函数R(s,a);
        # 4.折扣因子gamma;
        # 5.由上一次策略policy更新过的状态价值函数.
        for action in actions:

            # 通过先验的状态转移概率函数基于现有状态state和执行的动作action获得下一个状态next_state与状态转移概率
            next_state, transition_probability = state_transition_function(state, action)
            # 根据当前状态state与执行的动作action输入奖励函数获得此刻的即时奖励，本问题中需要用next_state计算即时奖励
            reward = get_reward(state, action, next_state)
            # 基于即时函数,折扣因子,状态转移概率,更新后的状态转移函数输出此时在状态s执行动作a的动作价值函数值
            action_value = reward + gamma * transition_probability * new_value_function_based_on_policy[next_state]
            # 选取在此状态s下所有可执行动作的动作价值最大的动作作为此状态下新的动作
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        # 更新策略
        new_policy[state] = best_action

    return new_policy

# step8.3: 构建策略迭代过程优化MDP输出最优策略(Policy)
def policy_iteration(policy, gamma, n, state_value_function=initialization_state_value_function):

    for i in range(n):
        # 策略评估(Policy Evaluation): 使用贝尔曼期望方程计算"当前策略"下状态空间中所有状态的价值函数值V_pi(s)
        new_state_value_function_based_on_policy = update_state_value_function_based_on_policy(policy, gamma, 100, state_value_function)

        # 策略改进(Policy Improvement): 根据状态价值函数更新策略，选择使得动作价值函数Q_pi(s,a)最大的动作
        new_policy = policy_improvement(new_state_value_function_based_on_policy, gamma)

        if new_policy == policy:
            break
        policy = new_policy

    return policy

best_policy = policy_iteration(policy, gamma, n_iterations)

# 打印最佳策略
print("最佳策略:")
for state in states:
    print(f"状态 {state}: 动作 {best_policy[state]}")

# 打印状态价值函数（可选）
value_function_final = update_state_value_function_based_on_policy(best_policy, gamma, 100, {state: 0 for state in states})
print("\n状态价值函数:")
for state in states:
    print(f"状态 {state}: 价值 {value_function_final[state]}")