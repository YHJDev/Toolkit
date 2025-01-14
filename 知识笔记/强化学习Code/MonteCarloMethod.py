# 例子1: 估算圆周率
# 步骤:
# step1随机变量识别:在单位正方形内随机选择点;
# step2构建概率分布模型:每个点的坐标均匀分布在 [0, 1] 范围内;
# step3生成随机样本:生成大量 (x, y) 坐标点;
# step4构建模型:统计落在单位圆内的点数;
# step5模型结果分析:用落在圆内的点数与总点数之比来估算 π 值.

import numpy as np

def estimate_pi(num_samples):
    # step1随机变量识别,在单位正方形内随机选择点
    inside_circle = 0

    for _ in range(num_samples):

        # step2构建概率分布模型,step3生成随机样本
        x, y = np.random.rand(2)
        # step4构建解决问题的模型
        if x**2 + y**2 <= 1:  # 判断是否在单位圆内
            inside_circle += 1
            # step5对模拟结果进行分析
    return (inside_circle / num_samples) * 4

pi_estimate = estimate_pi(10000)
print(f"估算的圆周率值: {pi_estimate}")

# 例子2:金融风险评估
# 步骤:
# step1随机变量识别:投资回报率、市场波动性等;
# step2构建概率分布模型:假设投资回报率服从正态分布，市场波动性服从均匀分布;
# step3生成随机样本:从这些分布中抽取投资回报率与市场波动性样本;
# step4构建模型:计算不同投资组合的回报和风险;
# step5模型结果分析：评估不同组合的期望收益和风险水平，绘制收益-风险图.

import numpy as np

def simulate_investment(num_simulations):
    # step1, step2, step3
    returns = np.random.normal(loc=0.08, scale=0.15, size=num_simulations)  # 假设年回报率
    risks = np.random.uniform(low=0.05, high=0.25, size=num_simulations)  # 假设市场波动性

    return returns, risks

returns, risks = simulate_investment(10000)

print("投资回报率随机样本:", returns)
print("市场波动性样本:", risks)
# 基于投资回报率样本与市场波动性样本进行进一步的期望收益和风险评估建模

# 例子3:排队系统分析
# 步骤:
# step1随机变量识别:顾客到达时间、服务时间等;
# step2构建概率分布模型:顾客到达时间服从泊松过程，服务时间服从指数分布;
# step3生成随机样本:模拟顾客到达和服务过程;
# step4模型构建:记录排队长度、等待时间等指标;
# step5模型结果分析:计算平均等待时间、系统利用率等.

import numpy as np

def queue_simulation(num_customers):
    # step1,step2,step3
    arrival_times = np.random.exponential(scale=1 / 2, size=num_customers)  # 到达时间
    service_times = np.random.exponential(scale=1 / 3, size=num_customers)  # 服务时间

    # step4:1个服务员串行服务num_customers个顾客,记录每个顾客需要等待服务的时间
    wait_times = []
    current_time = 0
    for arrival, service in zip(arrival_times, service_times):

        current_time += arrival  # 顾客到达
        wait_times.append(max(0, current_time - arrival))  # 等待时间
        current_time += service  # 服务结束

           # step5:求顾客平均需要等待的时间
    return np.mean(wait_times)

num_customers = 100
avg_wait_time = queue_simulation(num_customers)

print("顾客流量为:", num_customers, "平均等待时间:", avg_wait_time)