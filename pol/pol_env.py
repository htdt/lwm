import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
STEPS = [LEFT, DOWN, RIGHT, UP]
OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}


def step_grid(cur, d, size):
    x, y = cur
    if d == LEFT:
        x -= 1
    elif d == RIGHT:
        x += 1
    elif d == UP:
        y -= 1
    elif d == DOWN:
        y += 1
    if x < 0 or y < 0 or x >= size or y >= size:
        return cur
    return (x, y)


def gen_labyrinth(size, np_random):
    edges = np.zeros((size, size, 4), dtype=bool)
    visit = np.zeros((size, size), dtype=bool)
    stack = [(0, 0)]

    while len(stack):
        cur = stack.pop()
        visit[cur] = 1
        neib = [d for d in STEPS if not visit[step_grid(cur, d, size)]]
        if len(neib):
            stack.append(cur)
            next_d = np_random.choice(neib)
            next_pos = step_grid(cur, next_d, size)
            edges[cur][next_d] = edges[next_pos][OPPOSITE[next_d]] = 1
            stack.append(next_pos)
    return edges


class PolEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, size):
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(4)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action in STEPS
        if self.map[self.pos][action]:
            self.pos = step_grid(self.pos, action, self.size)
        self.visit[self.pos] = 1
        done = self.visit.all()
        state = self.map[self.pos].astype(np.uint8)
        reward = -1.0
        return state, reward, done, {}

    def reset(self):
        self.map = gen_labyrinth(self.size, self.np_random)
        self.visit = np.zeros((self.size, self.size), dtype=bool)
        # self.pos = (0, 0)
        self.pos = tuple(self.np_random.randint(self.size, size=2))
        self.visit[self.pos] = 1
        return self.map[self.pos].astype(np.uint8)

    def render(self, mode="human"):
        m2 = np.zeros((self.size * 2 + 1, self.size * 2 + 1), dtype=int)
        m2[1::2, 1::2] = 1
        m2[:-1:2, 1::2] = self.map[:, :, LEFT]
        m2[1::2, :-1:2] = self.map[:, :, UP]
        m2[self.pos[0] * 2 + 1, self.pos[1] * 2 + 1] = 2

        for s in m2.astype(str):
            s = "".join(s)
            s = s.replace("0", "#")
            s = s.replace("1", " ")
            s = s.replace("2", "@")
            sys.stdout.write(s + "\n")

    def close(self):
        self.map = None
