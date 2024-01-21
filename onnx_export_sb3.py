import torch as th
import os
import datetime
from stable_baselines3 import PPO, A2C, SAC


class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net, output_size):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
        version_number = th.Tensor([3])
        self.version_number = th.nn.Parameter(version_number, requires_grad=False)

        memory_size = th.Tensor([0])
        self.memory_size = th.nn.Parameter(memory_size, requires_grad=False)

        output_shape=th.Tensor([output_size])  # continuous_action_output_shape
        self.output_shape = th.nn.Parameter(output_shape, requires_grad=False)

    def forward(self, observation):
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.output_shape, self.version_number, self.memory_size

if __name__ == '__main__':
    model = PPO.load("models/torch/policy_networks/20230712_191458.009_model_SbPPO")
    
    # Enter the model name below
    model_name = "20230712_191458.009_model_SbPPO"
    loading_path = os.path.join("models/torch/policy_networks", model_name)
    if "A2C" in model_name:
        model = A2C.load(loading_path)
    elif "PPO" in model_name:
        model = PPO.load(loading_path)
    elif "SAC" in model_name:
        model = SAC.load(loading_path)
    
    onnxable_model = OnnxablePolicy(
    model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net, 3
    )

    os.makedirs("models/unity", exist_ok=True)
    date_string = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join("models/unity", date_string + "_SB3.onnx")

    dummy_input = th.randn(1, 48)
    # Export the model to ONNX
    th.onnx.export(
        onnxable_model,
        dummy_input,
        save_path,
        opset_version=11,
        input_names=["obs_0"], verbose=True,
        output_names = [ 'continuous_actions', 'continuous_action_output_shape', 
                        'version_number', 'memory_size'],
        dynamic_axes={'obs_0': {0: 'batch'},#'vector_observation': {0: 'batch'},
                    'continuous_actions': {0: 'batch'},
                    'continuous_action_output_shape': {0: 'batch'}}
    )
