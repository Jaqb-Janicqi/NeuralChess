import numpy as np
import torch
import resnet
import resnet_old
import torch_directml

device = torch_directml.device()
model_dict = torch.load("v_only.pth")
model = resnet.ResNet(10, 128, 9, 1968)
model.load_state_dict(model_dict["model_state_dict"])
model.half()
model.dtype = torch.float16
model.eval()
model.to(device)

# print dtype
print(model.start_block[0].weight.dtype)

import torch.nn.functional as F
data = np.zeros((9, 8, 8), dtype=np.float16)
data = model.to_tensor(data)
policy, val = model(data)
val = val.flatten()
policy = model.get_policy(policy)
val = model.get_value(val)
print(val)
print(policy.shape)

torch.save({
    'epoch': model_dict["epoch"],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': model_dict["optimizer_state_dict"],
    'log_dict': model_dict["log_dict"],
    'num_blocks': 10,
    'num_features': 128,
    'input_features': 9,
    'disable_policy': True,
    'squeeze_and_excitation': False,
    'policy_size': 1968,
}, f'model.pth')
