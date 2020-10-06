import torch
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.wrappers import TimeLimit
from pol_env import PolEnv


def make_vec_envs(num, size, seed=0, max_ep_len=1000):
    def make_env(rank):
        def _thunk():
            env = PolEnv(size)
            env = TimeLimit(env, max_episode_steps=max_ep_len)
            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            return env

        return _thunk

    envs = [make_env(i) for i in range(num)]
    envs = SubprocVecEnv(envs, context="fork")
    envs = VecTorch(envs)
    return envs


class VecTorch(VecEnvWrapper):
    def reset(self):
        return torch.from_numpy(self.venv.reset())

    def step_async(self, actions):
        assert len(actions.shape) == 2
        actions = actions[:, 0].cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs)
        reward = torch.from_numpy(reward)[..., None].float()
        done = torch.tensor(done, dtype=torch.uint8)[..., None]
        return obs, reward, done, info
