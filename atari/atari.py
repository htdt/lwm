import torch
from gym.spaces.box import Box
from baselines import bench
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.atari_wrappers import wrap_deepmind, make_atari
from baselines.common.vec_env import VecEnvWrapper


def make_vec_envs(name, num, seed=0, max_ep_len=100000):
    def make_env(rank):
        def _thunk():
            full_name = f"{name}NoFrameskip-v4"
            env = make_atari(full_name, max_episode_steps=max_ep_len)
            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            env = wrap_deepmind(env, episode_life=True, clip_rewards=False)
            return env

        return _thunk

    envs = [make_env(i) for i in range(num)]
    envs = ShmemVecEnv(envs, context="fork")
    envs = VecTorch(envs)
    return envs


class VecTorch(VecEnvWrapper):
    def __init__(self, env):
        super(VecTorch, self).__init__(env)
        obs = self.observation_space.shape
        self.observation_space = Box(0, 255, [obs[2], obs[0], obs[1]],
                                     dtype=self.observation_space.dtype)

    def _convert_obs(self, x):
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    def reset(self):
        return self._convert_obs(self.venv.reset())

    def step_async(self, actions):
        assert len(actions.shape) == 2
        actions = actions[:, 0].cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self._convert_obs(obs)
        reward = torch.from_numpy(reward)[..., None].float()
        done = torch.tensor(done, dtype=torch.uint8)[..., None]
        return obs, reward, done, info
