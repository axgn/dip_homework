import random

import numpy as np

# ----- 定义环境 -----


class GridWorld:
    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        # 边界判断
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            reward = -0.1
            next_state = self.state
        else:
            next_state = (x, y)
            reward = -0.01

        if next_state == self.goal:
            reward = 1.0

        self.state = next_state
        done = next_state == self.goal
        return next_state, reward, done


# ----- Q-learning -----

env = GridWorld()

q_table = np.zeros((4, 4, 4))  # (x, y, action)
alpha = 0.1  # 学习率
gamma = 0.98  # 折扣因子
epsilon = 0.2  # 探索概率

episodes = 500


def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)
    x, y = state
    return np.argmax(q_table[x, y])


for ep in range(episodes):
    state = env.reset()
    while True:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        x, y = state
        nx, ny = next_state

        # Q-learning 更新
        q_value = q_table[x, y, action]
        max_next_q = np.max(q_table[nx, ny])
        q_table[x, y, action] = q_value + alpha * (reward + gamma * max_next_q - q_value)

        state = next_state

        if done:
            break

print("训练完成！Q 表如下：")
print(q_table)

state = env.reset()
path = [state]

while state != env.goal:
    x, y = state
    action = np.argmax(q_table[x, y])  # 选最优动作
    next_state, reward, done = env.step(action)
    path.append(next_state)
    state = next_state

print("路径：", path)
