import numpy as np
import torch
import torch.distributions as distr
from lib.ModelType import TrainingData
from typing import List
from ptan.experience import ExperienceFirstLast


def unpack_batch_a2c(batch: List[ExperienceFirstLast], net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states: List[TrainingData] = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_vals_v = net(last_states)
        # print(f"last_states_v{last_states_v.shape}:", last_states_v)
        # print(f"last_vals_v{last_vals_v.shape}:", last_vals_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states, actions_v, ref_vals_v

def getMask(batch_data: List[TrainingData]):
    coords_mask_list = []
    for data in batch_data:
        _pre_features = torch.tensor(data.pre_features, dtype=torch.float32)
        coords_mask = _pre_features[:, -1]
        coords_mask_list.append(coords_mask.unsqueeze(0))
    return torch.cat(coords_mask_list)

@torch.no_grad()
def unpack_batch_sac(batch: List[ExperienceFirstLast], val_net, twinq_net, policy_net,
                     gamma: float, ent_alpha: float,
                     device="cpu"):
    """
    Unpack Soft Actor-Critic batch
    """
    states, actions_v, ref_q_v = \
        unpack_batch_a2c(batch, val_net, gamma, device)

    # references for the critic network
    mu_v = policy_net(states)
    act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    acts_v = act_dist.sample()
    
    q1_v, q2_v = twinq_net(states, acts_v)
    

    mask = getMask(states)

    log_prob = act_dist.log_prob(acts_v)
   
    ref_vals_v = torch.min(q1_v, q2_v) - \
                ent_alpha * torch.sum(log_prob.sum(-1) * mask, dim=-1, keepdim=True) / (torch.sum(mask, -1, keepdim=True) * log_prob.shape[-1])
   
    ref_vals_v = ref_vals_v.squeeze()
   
    return states, actions_v, ref_vals_v, ref_q_v

