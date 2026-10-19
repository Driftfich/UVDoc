from models import FastViTUNet
import torch
import time
from thop import profile

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = FastViTUNet(pretrained=False).to(device)
try:
    model.load_state_dict(torch.load("unet_bm_best.pth", map_location=device))
except FileNotFoundError:
    pass
model.eval()

# Dummy Input analog zur Inferenz (1 Batch, 5 Kanäle [RGB + Grid], 256x256)
dummy_input = torch.randn(1, 5, 256, 256).to(device)

# 1. Parameter
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f} M")

# 2. GFLOPs
macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
gflops = (macs * 2) / 1e9 # 1 MAC = 2 FLOPs
print(f"Compute: {gflops:.2f} GFLOPs")

# 3. Inferenzzeit messen
runs = 1000

# Warmup für realistische GPU-Zustände
with torch.no_grad():
    for _ in range(20):
        _ = model(dummy_input)

if device.type == 'cuda':
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            
    end_event.record()
    torch.cuda.synchronize()
    
    avg_time_ms = start_event.elapsed_time(end_event) / runs
else:
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    avg_time_ms = ((time.perf_counter() - start_time) * 1000) / runs

print(f"Avg Inference Time ({device}): {avg_time_ms:.2f} ms")