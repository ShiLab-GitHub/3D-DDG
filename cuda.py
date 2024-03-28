import torch
torch.cuda.current_device()
torch.cuda._initialized = True

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
