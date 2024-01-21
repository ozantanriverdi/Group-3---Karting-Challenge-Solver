import argparse
import torch
import os
import datetime

from networks.policy_network import PolicyNetwork as PolicyNetwork
from networks.policy_network_beta import PolicyNetwork as PolicyNetworkBeta

class KartingBrain(torch.nn.Module):
    def __init__(self, policy, output_size):
        super().__init__()
        self.policy = policy
        version_number = torch.Tensor([3])
        self.version_number = torch.nn.Parameter(version_number, requires_grad=False)

        memory_size = torch.Tensor([0])
        self.memory_size = torch.nn.Parameter(memory_size, requires_grad=False)

        output_shape=torch.Tensor([output_size])  # continuous_action_output_shape
        self.output_shape = torch.nn.Parameter(output_shape, requires_grad=False)

    def forward(self, observation):
        with torch.no_grad():
            mean, _ = self.policy(observation)


        return mean, self.output_shape, self.version_number, self.memory_size
    
class KartingBrainBeta(torch.nn.Module):
    def __init__(self, policy, output_size):
        super().__init__()
        self.policy = policy
        version_number = torch.Tensor([3])
        self.version_number = torch.nn.Parameter(version_number, requires_grad=False)

        memory_size = torch.Tensor([0])
        self.memory_size = torch.nn.Parameter(memory_size, requires_grad=False)

        output_shape=torch.Tensor([output_size])  # continuous_action_output_shape
        self.output_shape = torch.nn.Parameter(output_shape, requires_grad=False)

    def forward(self, observation):
        self.eval()
        with torch.no_grad():

            alpha, beta, mean, log_std = self.policy(observation)
        
            mean_alpha_beta = (alpha / (alpha + beta)) * 2 - 1

            mean_alpha_beta = mean_alpha_beta.view(1,-1)

            concat_mean = torch.cat((mean_alpha_beta, torch.tensor([[mean[0,0],mean[0,1]-0.2]])), dim=1)
        return concat_mean, self.output_shape, self.version_number, self.memory_size
    
if __name__ == '__main__':
    model = PolicyNetworkBeta(state_dim=48, action_dim=3) # adjsut if you want to export normal PolicyNetwork 
    model.load_checkpoint("20230710_190432.671_model")
    brain = KartingBrainBeta(model,3)

    os.makedirs("models/unity", exist_ok=True)
    date_string = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join("models/unity", date_string + ".onnx")
    print('---save_path',save_path)

    dummy_input = torch.randn(1, 48)
    torch.onnx.export(
        brain, dummy_input, save_path, opset_version=11, input_names=["obs_0"],
        verbose=True,
        output_names = [ 'continuous_actions', 'continuous_action_output_shape', 
                        'version_number', 'memory_size'],
        dynamic_axes={'obs_0': {0: 'batch'},
                    'continuous_actions': {0: 'batch'},
                    'continuous_action_output_shape': {0: 'batch'}}
)