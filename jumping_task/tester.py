import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, CLIPModel

from env_custom import JumpTaskEnvWithColors

from agents import AgentClip, AgentClipOnly, AgentClipDropout, AgentNormal

# env setup

envs = gym.make('jumping-colors-task-v0', scr_w=64, scr_h=64)
envs.single_action_space = envs.action_space
envs.is_vector_env = True

# agent = AgentNormal(envs).to('cuda')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
for param in clip_model.parameters():
  param.requires_grad = False
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
agent = AgentClip(envs, clip_model=clip_model, clip_processor=clip_processor, text="white rectangle near green square but not touching it.").to('cuda')

env = JumpTaskEnvWithColors(scr_w=64, scr_h=64)

averages = []


agent.load_state_dict(torch.load('clip.pt'))
agent.eval()


for obstacle_position in range(14, 48): # [14,48]
  for floor_height in range(0, 41): # [0,41]
    current_state, _ = env._reset(obstacle_position=obstacle_position, floor_height=floor_height)
    current_state = current_state.copy()
    current_state = torch.Tensor(current_state.copy()).to('cuda')
    done = False
    averages.append((obstacle_position,floor_height,True, 0))

    while not done:
      averages[-1] = (averages[-1][0],averages[-1][1],averages[-1][2],averages[-1][3]+1)
      
      action, logprob, _, value = agent.get_action_and_value(current_state)

      current_state, reward, done, _, info = env.step(action)
      current_state = torch.Tensor(current_state.copy()).to('cuda')

      if info['collision'] or reward==-1:
        averages[-1] = (obstacle_position,floor_height,False, averages[-1][3])
        break

all = len(averages)
pos = len([i for i in averages if i[2]])
neg = len([i for i in averages if not i[2]])
print('succesful for ',pos)
print('unsuccesful for ',neg)

pos = max([i[3] for i in averages if i[2]])
print(pos)