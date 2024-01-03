import torch

GPU_IDX = 2
DEVICE = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(DEVICE)

