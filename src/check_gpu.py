import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the device name
    device_name = torch.cuda.get_device_name(0)
    print(f"Success! CUDA is working.")
    print(f"Detected GPU: {device_name}")
else:
    print("GPU not found. PyTorch is running on CPU.")