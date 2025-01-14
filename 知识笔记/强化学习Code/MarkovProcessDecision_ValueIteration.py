import numpy as np

# 问题定义:
# 在一个3x3的空间里，一个小球在最左上角，需要找到小球的最优行动策略到空间里的最右下角。
# 小球每次移动一格，动作包括上下左右移动。
# 定义状态空间为:小球所在的空间位置。
# 定义动作空间为:小球可以上下左右移动。
# 定义状态转移函数为:当执行动作时，若未碰到墙壁，小球移动到相应的下一个状态，否则停留在原地。
# 定义即时奖励函数为:小球到达指定地点（2,2）时获得+10奖励，碰墙时获得-1奖励，其他情况下奖励为0。
# 定义折扣因子 gamma = 0.9。

# step1:构建状态空间 S
states = [(i, j) for i in range(3) for j in range(3)]

# step2:初始化状态价值函数 V(s)
# 使用零值初始化所有状态的价值函数
state_value_function = {state: 0 for state in states}

# step3:构建动作空间 A
actions = ['up', 'down', 'left', 'right']

# step4:定义折扣因子 gamma
gamma = 0.9

# step5:定义最大迭代次数和阈值以判断收敛
max_iterations = 1000

# step6:定义状态价值函数收敛阈值
threshold = 1e-6

# step7:定义奖励函数 R(s, a, s')
def get_reward(state, action, next_state):

    if next_state == (2, 2):
        return 10  # 到达目标位置获得奖励+10
    elif state == next_state:
        return -1  # 碰到墙壁获得惩罚-1
    else:
        return 0   # 其他情况下奖励为0

# step8:定义状态转移函数 T(s, a)
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

# step9:实现值迭代算法以优化 MDP，找到最优状态价值函数 V(s)
def value_iteration(states, actions, gamma, max_iterations, threshold, state_value_function):

    for iteration in range(max_iterations):

        max_delta = 0  # 记录价值函数的最大变化量
        next_state_value_function = state_value_function.copy()

        # 对于每一个状态，更新其价值函数
        for state in states:

            if state == (2, 2):
                # 目标状态的价值固定为0
                continue
            max_action_value = float('-inf')

            # 计算此state所有可能的动作的q
            for action in actions:

                # 获取下一个状态
                next_state, transition_probability = state_transition_function(state, action)
                # 获取即时奖励
                reward = get_reward(state, action, next_state)
                # 计算最优动作价值（由于确定性，概率为1）
                action_value = reward + gamma * transition_probability * state_value_function[next_state]
                # 选择最大的动作价值
                if action_value > max_action_value:
                    max_action_value = action_value

            # 使用最大动作价值函数值更新状态价值函数
            next_state_value_function[state] = max_action_value

            # 计算更新后状态价值函数与更新前状态价值函数值最大变化量,为收敛性判断做准备
            max_delta = max(max_delta, abs(next_state_value_function[state] - state_value_function[state]))

        state_value_function = next_state_value_function.copy()

        # 检查更新后的状态价值函数是否收敛
        if max_delta < threshold:
            print(f"值迭代在第 {iteration + 1} 次迭代后收敛。")
            break

    return state_value_function

# step10:根据最优状态价值函数V(s) 提取最优策略 π(s)
def extract_policy(states, actions, optimal_state_value_function, gamma):

    policy = {}
    for state in states:
        if state == (2, 2):
            # 目标状态，无需动作
            policy[state] = None
            continue

        max_value = float('-inf')
        best_action = None
        # 考虑此状态所有可能执行的动作，使用动作价值函数选择使动作价值函数最大的动作
        for action in actions:
            next_state,  transition_probability = state_transition_function(state, action)
            reward = get_reward(state, action, next_state)
            # 动作价值函数
            action_value = reward + gamma * optimal_state_value_function[next_state]

            if action_value > max_value:
                max_value = action_value
                best_action = action

        policy[state] = best_action

    return policy

# step11:运行值迭代算法并提取最优策略
optimal_value_function = value_iteration(states, actions, gamma, max_iterations, threshold, state_value_function)
optimal_policy = extract_policy(states, actions, optimal_value_function, gamma)

# 打印最优策略
print("最优策略:")
for state in states:
    print(f"状态 {state}: 动作 {optimal_policy[state]}")

# 打印最优状态价值函数
print("\n最优状态价值函数:")
for state in states:
    print(f"状态 {state}: 价值 {optimal_value_function[state]:.2f}")
