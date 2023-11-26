import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, CLIPModel

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class AgentClip(nn.Module):
    def __init__(self, envs, clip_model, clip_processor):
        super().__init__()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        self.clip_network = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )

    def get_value(self, x):
        with torch.no_grad():
            inputs = self.clip_processor(text=["a maze with the white square near the yellow square."], images=x, return_tensors="pt", padding=True).to('cuda')
            outputs = self.clip_model(**inputs)
            img_embeds = outputs.image_embeds # batch_size * 512
            text_embeds = outputs.text_embeds.repeat(img_embeds.shape[0],1) # batch_size * 512

        embed_clip = torch.cat((img_embeds, text_embeds), dim=1)
        embed_clip = self.clip_network(embed_clip)
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw" batch_size * 256
        hidden = torch.cat((hidden, embed_clip), dim=1)
        return self.critic(hidden)  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        with torch.no_grad():
            inputs = self.clip_processor(text=["a maze with the white square near the yellow square."], images=x, return_tensors="pt", padding=True).to('cuda')
            outputs = self.clip_model(**inputs)
            img_embeds = outputs.image_embeds # batch_size * 512
            text_embeds = outputs.text_embeds.repeat(img_embeds.shape[0],1) # batch_size * 512

        embed_clip = torch.cat((img_embeds, text_embeds), dim=1)
        embed_clip = self.clip_network(embed_clip)
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw" batch_size * 256
        hidden = torch.cat((hidden, embed_clip), dim=1)
      
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
class AgentNormal(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class AgentClipOnly(nn.Module):
    def __init__(self, envs, clip_model, clip_processor):
        super().__init__()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

        self.network = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
        )

    def get_value(self, x):
        with torch.no_grad():
            inputs = self.clip_processor(text=["a maze with the white square near the yellow square."], images=x, return_tensors="pt", padding=True).to('cuda')
            outputs = self.clip_model(**inputs)
            img_embeds = outputs.image_embeds # batch_size * 512
            text_embeds = outputs.text_embeds.repeat(img_embeds.shape[0],1) # batch_size * 512

        embed_clip = torch.cat((img_embeds, text_embeds), dim=1)
        hidden = self.network(embed_clip)
        return self.critic(hidden)  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        with torch.no_grad():
            inputs = self.clip_processor(text=["a maze with the white square near the yellow square."], images=x, return_tensors="pt", padding=True).to('cuda')
            outputs = self.clip_model(**inputs)
            img_embeds = outputs.image_embeds # batch_size * 512
            text_embeds = outputs.text_embeds.repeat(img_embeds.shape[0],1) # batch_size * 512

        embed_clip = torch.cat((img_embeds, text_embeds), dim=1)
        hidden = self.network(embed_clip)
      
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)