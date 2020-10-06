import sys
import torch
import time
from gym.envs.atari.atari_env import ACTION_MEANING
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from atari import make_vec_envs
from dqn.model import mnih_cnn


ACTION_ID = {v: k for k, v in ACTION_MEANING.items()}
KEY2ACTION = {
    "w": ACTION_ID["UP"],
    "s": ACTION_ID["DOWN"],
    "a": ACTION_ID["LEFT"],
    "d": ACTION_ID["RIGHT"],
    "f": ACTION_ID["FIRE"],
    "i": ACTION_ID["UPFIRE"],
    "k": ACTION_ID["DOWNFIRE"],
    "j": ACTION_ID["LEFTFIRE"],
    "l": ACTION_ID["RIGHTFIRE"],
    "u": ACTION_ID["UPLEFTFIRE"],
    "o": ACTION_ID["UPRIGHTFIRE"],
}


def convert_key(a):
    return KEY2ACTION.get(chr(a), ACTION_ID["NOOP"])


def key_press(key, mod):
    global cur_action, restart, pause
    if key == 0xFF0D:
        restart = True
    if key == 32:
        pause = not pause
    cur_action = convert_key(key)


def key_release(key, mod):
    global cur_action
    a = convert_key(key)
    if cur_action == a:
        cur_action = 0


def plot(x, fname):
    path2d = TSNE().fit_transform(x)
    x, y = tuple(zip(*path2d))
    plt.figure(figsize=(12, 12))

    plt.scatter(x, y, c=list(range(len(x))), alpha=0.5, s=200)
    plt.savefig(fname)
    for i in range(0, len(x), 10):
        plt.text(
            x[i],
            y[i],
            str(i),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )
    plt.savefig("num_" + fname)


if __name__ == "__main__":
    name = "MontezumaRevenge" if len(sys.argv) < 2 else sys.argv[1]
    env = make_vec_envs(name, 1)

    conv_wmse = mnih_cnn(1, 32)
    conv_idf = mnih_cnn(1, 32)
    conv_cpc = mnih_cnn(1, 32)
    conv_rnd = mnih_cnn(1, 32)
    conv_wmse.load_state_dict(torch.load("models/conv_wmse.pt", map_location="cpu"))
    conv_idf.load_state_dict(torch.load("models/conv_idf.pt", map_location="cpu"))
    conv_cpc.load_state_dict(torch.load("models/conv_cpc.pt", map_location="cpu"))
    conv_wmse.eval(), conv_idf.eval(), conv_cpc.eval(), conv_rnd.eval()

    mem = torch.empty(4, 1000, 32)
    cursor = 0

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    window_still_open = True

    while window_still_open:
        cur_action = 0
        restart = False
        pause = False

        obs = env.reset()
        total_reward = steps = 0
        while 1:

            steps += 1
            if steps == 1000:
                break
            a = torch.tensor([[cur_action]])
            obs, r, done, info = env.step(a)

            with torch.no_grad():
                obs = obs.float() / 128 - 1
                mem[0, cursor] = conv_wmse(obs)[0]
                mem[1, cursor] = conv_idf(obs)[0]
                mem[2, cursor] = conv_cpc(obs)[0]
                mem[3, cursor] = conv_rnd(obs)[0]

            if cursor % 10 == 0:
                print(cursor)
            cursor += 1

            if r != 0:
                print(f"reward {r.item():0.2f}")
            total_reward += r.item()
            window_still_open = env.render()
            if not window_still_open or done or restart:
                break
            while pause:
                env.render()
                time.sleep(0.1)
            time.sleep(0.1)

        print(f"timesteps {steps} reward {total_reward:0.2f}")

    plot(mem[0, :cursor], "plot_wmse.png")
    plot(mem[1, :cursor], "plot_idf.png")
    plot(mem[2, :cursor], "plot_cpc.png")
    plot(mem[3, :cursor], "plot_rnd.png")
