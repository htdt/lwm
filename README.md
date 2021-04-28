# Latent World Models For Intrinsically Motivated Exploration
Official repository | [arXiv:2010.02302](https://arxiv.org/abs/2010.02302) | NeurIPS 2020 Spotlight

[10m video presentation from NeurIPS](https://slideslive.com/38937965/latent-world-models-for-intrinsically-motivated-exploration)

![montezuma's revenge t-sne](https://raw.githubusercontent.com/htdt/lwm/master/montezuma.png)

## Installation
The implementation is based on PyTorch. Logging works on [wandb.ai](https://wandb.ai/). See `docker/Dockerfile`.

## Usage
After training, the resulting models will be saved as `models/dqn.pt`, `models/predictor.pt` etc.
For evaluation, models will be loaded from the same filenames.

#### Atari
To reproduce LWM results from [Table 2](https://arxiv.org/abs/2010.02302):
```sh
cd atari
python -m train --env MontezumaRevenge --seed 0
python -m eval --env MontezumaRevenge --seed 0
```
See `default.yaml` for detailed configuration.

To get trajectory plots as on Figure 3:
```sh
cd atari
# first train encoders for random agent
python -m train_emb
# next play the game with keyboard
python -m emb_vis
# see plot_*.png
```

#### Partially Observable Labyrinth
To reproduce scores from Table 1:
```sh
cd pol
# DQN agent
python -m train --size 3
python -m eval --size 3

# DQN + WM agent
python -m train --size 3 --add_ri
python -m eval --size 3 --add_ri

# random agent
python -m eval --size 3 --random
```

Code of the environment is in [pol/pol_env.py](https://github.com/htdt/lwm/blob/master/pol/pol_env.py), it extends `gym.Env` and can be used as usual:
```python
from pol_env import PolEnv
env = PolEnv(size=3)
obs = env.reset()
action = env.observation_space.sample()
obs, reward, done, infos = env.step(action)
env.render()
#######
# #   #
# ### #
# #@  #
# # # #
#   # #
#######
```

## Bibtex
```
@inproceedings{LWM,
 author = {Ermolov, Aleksandr and Sebe, Nicu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {5565--5575},
 publisher = {Curran Associates, Inc.},
 title = {Latent World Models For Intrinsically Motivated Exploration},
 volume = {33},
 year = {2020}
}
```
