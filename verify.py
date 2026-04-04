import torch

print("--- Hardware Check ---")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# Check if it's the Nightly build (usually has a '+dev' or date in the version)
if 'dev' in torch.__version__ or 'nightly' in torch.__version__:
    print("Build Type: Nightly (Correct for Blackwell)")
else:
    print("Build Type: Stable (Note: May have issues with SM 120)")

# Test Compute
try:
    x = torch.ones(1, 1).cuda()
    print("Compute Test: PASSED")
except Exception as e:
    print(f"Compute Test: FAILED - {e}")

# Check VRAM
free, total = torch.cuda.mem_get_info()
print(f"VRAM: {free/1024**3:.2f}GB free / {total/1024**3:.2f}GB total")