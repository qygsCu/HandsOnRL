import numpy as np
from collections import defaultdict

class MCAgent:
    """ 蒙特卡洛智能体框架 """
    def __init__(self, action_space_size, epsilon=0.1, gamma=0.9):
        self.action_space_size = action_space_size # 动作空间大小
        self.epsilon = epsilon                     # 探索率
        self.gamma = gamma                         # 折扣因子
        
        # 使用 defaultdict，当遇到未见过的状态时，默认返回全是 0.0 的数组（注意这里是浮点数 0.0！）
        self.Q_table = defaultdict(lambda: np.zeros(action_space_size, dtype=float))
        
        # 记录每个 (状态, 动作) 对的累积回报之和以及被访问的次数，用于计算平均值
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float) # 使用 float 防止整数除法截断

    def get_action(self, state):
        """
        TODO 1: 实现 epsilon-greedy 策略
        - 以 epsilon 的概率随机选择一个动作
        - 以 1-epsilon 的概率选择当前 Q_table[state] 中值最大的动作
        """
        # --- 请在此处填充你的代码 ---
        r = np.random.rand()
        if r > self.epsilon:
            return np.argmax(self.Q_table[state])
        else:
            return np.random.randint(self.action_space_size)
        
        

    def update(self, trajectory):
        """
        TODO 2: 根据一条完整的轨迹（Episode）更新 Q_table
        - trajectory 是一个列表，里面按时间顺序存储了元组: [(state_0, action_0, reward_1), (state_1, action_1, reward_2), ...]
        - 你需要从后往前（反向遍历）计算真实的回报 G_t
        - 实现“首次访问（First-Visit）”的逻辑
        - 更新 self.returns_sum, self.returns_count, 以及 self.Q_table
        """
        G = 0.0  # 初始回报
        
        # 我们需要判断“首次访问”，所以先提取出轨迹中所有的 (state, action) 对
        visited_state_actions = [(step[0], step[1]) for step in trajectory]
        for i in reversed(range(len(trajectory))):
            state, action, r = trajectory[i]
            G = self.gamma * G + r
            if (state, action) not in visited_state_actions[:i]:
                self.returns_sum[(state, action)] += 1
                self.returns_count[(state, action)] += G
                self.Q_table[state][action] = self.returns_count[(state, action)] / self.returns_sum[(state, action)]


# ==========================================
# 下面是测试与运行脚手架（不需要修改）
# ==========================================

class DummyEnv:
    """ 一个极其简化的假环境，用来跑通你的代码逻辑 """
    def __init__(self):
        self.action_space_n = 2
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        return "State_A" # 始终从同一个状态开始
        
    def step(self, action):
        self.step_count += 1
        # 瞎编的转移逻辑：如果在 State_A 选动作 1，能拿到高分并结束
        if action == 1:
            return "State_B", 10.0, True 
        else:
            return "State_A", -1.0, (self.step_count >= 5)

def run_episode(env, agent):
    """ 运行一个完整的回合（Episode），收集轨迹数据 """
    trajectory = []
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        
    return trajectory

def main():
    env = DummyEnv()
    agent = MCAgent(action_space_size=env.action_space_n, epsilon=0.1, gamma=0.9)
    
    num_episodes = 1000
    print("开始训练...")
    
    for i in range(num_episodes):
        # 1. 采样一整条轨迹
        trajectory = run_episode(env, agent)
        # 2. 用这条轨迹进行事后更新
        agent.update(trajectory)
        
    print("训练结束！最终的 Q 表：")
    for state, q_values in agent.Q_table.items():
        print(f"{state}: {q_values}")

if __name__ == "__main__":
    main()