import random
from enum import Enum

import numpy as np
import pygame


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def render_ascii_with_arrows(env, path):
    grid = [[" . " for _ in range(env.col)] for _ in range(env.row)]

    for x, y in env.obstacles:
        grid[x][y] = " # "

    arrow = {(1, 0): "↓", (-1, 0): "↑", (0, 1): "→", (0, -1): "←"}

    for i in range(len(path) - 1):
        x, y = path[i]
        nx, ny = path[i + 1]

        dx, dy = nx - x, ny - y
        grid[x][y] = f" {arrow[(dx, dy)]} "

    sx, sy = env.start
    gx, gy = env.goal
    grid[sx][sy] = " S "
    grid[gx][gy] = " G "

    print("\n机器人的运动路径:")
    for row in grid:
        print("".join(row))
    print()


# ----- ASCII：只显示环境 -----


def render_env_only(env):
    grid = [[" . " for _ in range(env.col)] for _ in range(env.row)]

    # 障碍
    for x, y in env.obstacles:
        grid[x][y] = " # "

    # 起点终点
    sx, sy = env.start
    gx, gy = env.goal
    grid[sx][sy] = " S "
    grid[gx][gy] = " G "

    print("\n机器人当前的环境:")
    for row in grid:
        print("".join(row))
    print()


class GridWorld:
    def __init__(self, col=4, row=4, start=(0, 0), goal=(3, 3)):
        self.col = col
        self.row = row
        self.start = start
        self.goal = goal
        self.state = self.start
        self.obstacles = set()
        self.generate_obstacles()

    def generate_obstacles(self):
        n = random.randint(min(self.row, self.col) // 2, min(self.row, self.col) + 2)
        while len(self.obstacles) < n:
            x = random.randint(0, self.row - 1)
            y = random.randint(0, self.col - 1)
            if (x, y) != self.start and (x, y) != self.goal:
                self.obstacles.add((x, y))

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action: Action):
        x, y = self.state
        if action is Action.LEFT:
            y -= 1
        elif action is Action.RIGHT:
            y += 1
        elif action is Action.UP:
            x -= 1
        elif action is Action.DOWN:
            x += 1
        # 边界判断
        if x < 0 or x >= self.row or y < 0 or y >= self.col:
            reward = -0.1
            next_state = self.state
        else:
            next_state = (x, y)
            # reward = -0.01
            if next_state in self.obstacles:
                reward = -0.5
                next_state = self.state
            else:
                reward = -0.01

        if next_state == self.goal:
            reward = 1.0

        self.state = next_state
        done = next_state == self.goal
        return next_state, reward, done


# ----- Q-learning -----

row = 10
col = 10
start = (0, 0)

env = GridWorld(col=col, row=row, start=start, goal=(row - 1, col - 1))
render_env_only(env)
q_table = np.zeros((row, col, 4))  # (x, y, action)
alpha = 0.1
gamma = 0.98
epsilon = 0.2

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
        next_state, reward, done = env.step(Action(action))

        x, y = state
        nx, ny = next_state

        q_value = q_table[x, y, action]
        max_next_q = np.max(q_table[nx, ny])
        q_table[x, y, action] = q_value + alpha * (reward + gamma * max_next_q - q_value)

        state = next_state

        if done:
            break

# print("训练完成！Q 表如下：")
# print(q_table)

state = env.reset()
path = [state]
while state != env.goal:
    x, y = state
    action = np.argmax(q_table[x, y])
    next_state, reward, done = env.step(Action(action))
    path.append(next_state)
    state = next_state

print("路径：", path)
render_ascii_with_arrows(env, path)

# CELL_SIZE = 100
# MARGIN = 2
# ARROW_COLOR = (255, 0, 0)
# OBSTACLE_COLOR = (0, 0, 0)
# START_COLOR = (0, 255, 0)
# GOAL_COLOR = (0, 0, 255)
# PATH_COLOR = (255, 200, 0)
# BG_COLOR = (200, 200, 200)

# pygame.init()
# screen = pygame.display.set_mode((col*CELL_SIZE, row*CELL_SIZE))
# pygame.display.set_caption("GridWorld Q-learning")

# def draw_env(env, path=None):
#     screen.fill(BG_COLOR)
#     for i in range(env.row):
#         for j in range(env.col):
#             rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE-MARGIN, CELL_SIZE-MARGIN)
#             pygame.draw.rect(screen, (255, 255, 255), rect)
#             if (i, j) in env.obstacles:
#                 pygame.draw.rect(screen, OBSTACLE_COLOR, rect)
#             if (i, j) == env.start:
#                 pygame.draw.rect(screen, START_COLOR, rect)
#             if (i, j) == env.goal:
#                 pygame.draw.rect(screen, GOAL_COLOR, rect)
#     # 画路径箭头
#     if path:
#         for k in range(len(path)-1):
#             x1, y1 = path[k]
#             x2, y2 = path[k+1]
#             start_px = (y1*CELL_SIZE + CELL_SIZE//2, x1*CELL_SIZE + CELL_SIZE//2)
#             end_px = (y2*CELL_SIZE + CELL_SIZE//2, x2*CELL_SIZE + CELL_SIZE//2)
#             pygame.draw.line(screen, ARROW_COLOR, start_px, end_px, 5)
#             # 箭头
#             dx, dy = end_px[0]-start_px[0], end_px[1]-start_px[1]
#             angle = np.arctan2(dy, dx)
#             arrow_size = 15
#             pygame.draw.polygon(screen, ARROW_COLOR, [
#                 (end_px[0], end_px[1]),
#                 (end_px[0]-arrow_size*np.cos(angle-np.pi/6), end_px[1]-arrow_size*np.sin(angle-np.pi/6)),
#                 (end_px[0]-arrow_size*np.cos(angle+np.pi/6), end_px[1]-arrow_size*np.sin(angle+np.pi/6))
#             ])

# def draw_env_only(env):
#     draw_env(env, path=None)

# # ----- 动态展示路径 -----
# agent_state = env.reset()
# running = True
# clock = pygame.time.Clock()
# path_index = 0
# dynamic_path = [agent_state]

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     screen.fill(BG_COLOR)
#     draw_env(env)
#     # 动态更新路径
#     if path_index < len(path)-1:
#         next_state = path[path_index+1]
#         dynamic_path.append(next_state)
#         path_index += 1

#     draw_env(env, path=dynamic_path)
#     pygame.display.flip()
#     clock.tick(2)  # 每秒显示2帧

# pygame.quit()
