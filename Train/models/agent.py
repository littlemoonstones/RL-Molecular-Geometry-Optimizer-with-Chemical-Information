import ptan
import numpy as np

class AgentA2C(ptan.agent.BaseAgent):
	def __init__(self, net, device="cpu"):

		'''
		Defines the Agent Class. Purpose in life of this is to compute the actions given the states using the actor network
		and store the hidden states of actor and critic rnn layers and the action and value function tensors which will be
		handy while training.

		Inputs:
			-net: Actor network
			-critic : Critic Network
		'''
		self.net = net
		self.device = device

	def __call__(self, states, agent_states = None):
		states_v = ptan.agent.float32_preprocessor(states).to(self.device)
		#states_v = states_v.squeeze(dim = 0)
		if len(states_v.shape) == 2:
			states_v = states_v.unsqueeze(dim = 0)
		#Computing the mean of the action value given the input states

		mu_v = self.net(states_v)

		#Gym accepts the actions input as numpy arrays, hence converting the obtained tensor to numpy array
		mu = mu_v.squeeze().data.cpu().numpy()
		logstd = self.net.logstd.data.cpu().numpy()

		'''
        Action value for each atomic coordinate is a vector of size 2, 
        the first element contains the log of the magnitude of action
        and the second element contains the probability that the sign of the action is positive

        Hence the first element corresponds to a continuous action space while second to a discrete space.
        We are sampling the value for the first element using a gaussian (Reparametrization trick)
        For the second element, we are currently directy using the predicted mean value, but ideally we should sample from a bernoulli distribution.
        '''

		# actions = np.zeros_like(np.array([mu]))
		actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
		actions = np.clip(actions, -1, 1)

		return actions, states_v

	def reset(self):
		self.actor_hidden_states = None
