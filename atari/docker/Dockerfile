FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get install -y libgtk2.0-dev
RUN pip install opencv-python
RUN pip install tensorflow-gpu==1.14
RUN pip install git+https://github.com/openai/baselines
RUN pip install gym[atari]
RUN pip install wandb
RUN pip install ipdb
ENTRYPOINT wandb login $WANDB_KEY && /bin/bash
