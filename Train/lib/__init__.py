from typing import Dict, List
import ptan
import numpy as np
import torch
import math
import logging
import models.model as model

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def make_logger(name, log_file, level=logging.INFO, mode='a'):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode=mode)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def test_net(net: model.ModelActor, env, test_file: Dict[str, List[np.ndarray]], device="cpu"):
    rewards = 0.0
    steps = 0
    count = 0
    net.eval()
    for smile, coords in test_file.items():
        for coord in coords:
            count += 1
            obs = env.reset(smile = smile, coord = coord)
            while True:
                obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
                mu_v = net(obs_v)[0]
                action = mu_v.squeeze(dim=0).data.cpu().numpy()
                action = np.clip(action, -1, 1)
                if np.isscalar(action): 
                    action = [action]
                obs, reward, done, _ = env.step(action)
                rewards += reward
                steps += 1
                if done:
                    break
    net.train()
    return rewards / count, steps / count

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



