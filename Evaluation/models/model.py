import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.self_attn import MultiHeadAttention, ResLayerNorm
import ptan
import numpy as np

# test: hidden = 9, head = 1
class ModelActor(nn.Module):
    def __init__(self, input_size, act_size, hidden_size = 48, n_head = 3):
        super(ModelActor, self).__init__()
        self.head_num = n_head
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mh_self_attn1 = MultiHeadAttention(hidden_size, self.head_num)
        self.mh_addnorm1 = ResLayerNorm(hidden_size) 
        self.output_layer = nn.Linear(hidden_size, act_size)
        # self.logstd = nn.Parameter(torch.tensor([-2., -12.]), requires_grad = True)
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x: torch.Tensor):
        # if x.dim() == 3:
        #     x = x.squeeze()
        x, mask = x[:, :, :-1], x[:, :, -1]
        out = F.relu(self.input_layer(x))
        out = self.mh_addnorm1(out, self.mh_self_attn1, mask)
        out = self.output_layer(out)

        out = torch.tanh(out)
        # out = torch.clip(out, -1, 1)
        # print(out.dim(), out)
        # print("actor out", out)
        # raise Exception
        
        return out

class ModelCritic(nn.Module):
    def __init__(self, input_size, hidden_size = 48, n_head = 3):
        super(ModelCritic, self).__init__()
        self.head_num = n_head
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mh_self_attn1 = MultiHeadAttention(hidden_size, self.head_num)
        self.mh_addnorm1 = ResLayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, mask = x[:, :, :-1], x[:, :, -1]
        out = F.relu(self.input_layer(x))
        out = self.mh_addnorm1(out, self.mh_self_attn1, mask)
        out = self.output_layer(out)
        out = torch.sum(out*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)
        # print(out)
        # raise Exception("ModelCritic")
        return out

class ModelSACTwinQ(nn.Module):
    def __init__(self, input_size, act_size, hidden_size = 48, n_head = 3):
        super(ModelSACTwinQ, self).__init__()
        self.head_num = n_head

        self.input_layer1 = nn.Linear(input_size + act_size, hidden_size)
        self.mh_self_attn1 = MultiHeadAttention(hidden_size, self.head_num)
        self.mh_addnorm1 = ResLayerNorm(hidden_size)
        self.output_layer1 = nn.Linear(hidden_size, 1)

        self.input_layer2 = nn.Linear(input_size + act_size, hidden_size)
        self.mh_self_attn2 = MultiHeadAttention(hidden_size, self.head_num)
        self.mh_addnorm2 = ResLayerNorm(hidden_size)
        self.output_layer2 = nn.Linear(hidden_size, 1)

    def forward(self, obs, act):
        # print(obs.shape, act.shape)
        obs, mask = obs[:, :, :-1], obs[:, :, -1]

        # question
        x1 = torch.cat([obs, act], dim=-1)
        x2 = torch.cat([obs, act], dim=-1)
        # print(f"obs {obs.shape}", obs)
        # print(f"act {act.shape}", act)
        # print(f"x1 {x1.shape}", x1)
        # raise Exception

        out1 = F.relu(self.input_layer1(x1))
        out1 = self.mh_addnorm1(out1, self.mh_self_attn1, mask)
        # print(f"mh_addnorm1 {out1.shape}", out1)
        out1 = self.output_layer1(out1)
        # print(f"output_layer1 {out1.shape}", out1)
        out1 = torch.sum(out1*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)
        # print(f"out1 {out1.shape}", out1)
        # raise Exception

        
        out2 = F.relu(self.input_layer2(x2))
        out2 = self.mh_addnorm2(out2, self.mh_self_attn2, mask)
        out2 = self.output_layer2(out2)
        out2 = torch.sum(out2*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)
        # print("out1", out1)
        # raise Exception(self.__class__.__name__)
        return out1, out2

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states