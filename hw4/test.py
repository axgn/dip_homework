import sys

import numpy as np
import pygame

# --------- GridWorld 定义 ---------


class GridWorld:
    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = {(1, 2)}  # 例子：放一个障碍物
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        i, j = self.state

        if action == 0:  # up
            i -= 1
        elif action == 1:  # down
            i += 1
        elif action == 2:  # left
            j -= 1
        elif action == 3:  # right
            j += 1

        # 边界判定
        if not (0 <= i < self.size and 0 <= j < self.size):
            return self.state, -0.1, False

        # 障碍物判定
        if (i, j) in self.obstacles:
            return self.state, -0.1, False

        next_state = (i, j)
        reward = 1.0 if next_state == self.goal else -0.01
        self.state = next_state
        done = next_state == self.goal
        return next_state, reward, done


# --------- Pygame 显示 ---------

CELL = 80
GRID_COLOR = (200, 200, 200)
AGENT_COLOR = (50, 150, 255)
GOAL_COLOR = (255, 200, 0)
OB_COLOR = (80, 80, 80)
BG = (240, 240, 240)

pygame.init()
font = pygame.font.SysFont(None, 32)

env = GridWorld()
size = env.size

screen = pygame.display.set_mode((CELL * size, CELL * size))
pygame.display.set_caption("GridWorld 可视化")


def draw_grid(state):
    screen.fill(BG)

    for i in range(size):
        for j in range(size):
            rect = pygame.Rect(j * CELL, i * CELL, CELL, CELL)

            # 目标格
            if (i, j) == env.goal:
                pygame.draw.rect(screen, GOAL_COLOR, rect)

            # 障碍物
            elif (i, j) in env.obstacles:
                pygame.draw.rect(screen, OB_COLOR, rect)

            pygame.draw.rect(screen, GRID_COLOR, rect, 2)

    # agent
    ai, aj = state
    agent_rect = pygame.Rect(aj * CELL, ai * CELL, CELL, CELL)
    pygame.draw.rect(screen, AGENT_COLOR, agent_rect)

    pygame.display.flip()


# --------- 主循环 ---------

state = env.reset()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # 按键控制 agent
        if event.type == pygame.KEYDOWN:
            action = None
            if event.key == pygame.K_UP:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            elif event.key == pygame.K_LEFT:
                action = 2
            elif event.key == pygame.K_RIGHT:
                action = 3

            if action is not None:
                state, reward, done = env.step(action)
                print(f"state={state}, reward={reward}")

                if done:
                    print("到达目标，重置")
                    state = env.reset()

    draw_grid(state)
