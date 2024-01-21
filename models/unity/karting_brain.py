import numpy
import torch as th
from torch.distributions import MultivariateNormal


class KartingBrain(th.nn.Module):
  def __init__(self, policy, output_size):
    super().__init__()
    self.policy = policy
    version_number = th.Tensor([3])  # version_number
    self.version_number = th.nn.Parameter(version_number, requires_grad=False)

    memory_size = th.Tensor([0])  # memory_size
    self.memory_size = th.nn.Parameter(memory_size, requires_grad=False)

    output_shape=th.Tensor([output_size])  # continuous_action_output_shape
    self.output_shape = th.nn.Parameter(output_shape, requires_grad=False)

  def forward(self, observation):
    # TODO: add your own forward logic
    state_arr = numpy.array(observation)
    state = th.tensor(state_arr)
    mean, log_std = self.policy(state)
    std = th.exp(log_std)
    

    # normal = MultivariateNormal(mean, th.diag_embed(std))
    # actions = normal.sample()
    

  #  actions  = self.policy.get_distribution(observation).get_actions(deterministic=False)
    return actions, self.output_shape, self.version_number, self.memory_size