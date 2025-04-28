import torch
import torch_pruning as tp
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.pt').model
model.eval()

# Tạo input mẫu
example_inputs = torch.randn(1, 3, 640, 640).to(next(model.parameters()).device)

# Tạo dependency graph
ignored_layers = []  # có thể để trống
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=example_inputs)

# Tạo PRUNER mới
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs=example_inputs,
    importance=tp.importance.MagnitudeImportance(p=2),  # L2 Norm
    iterative_steps=1,
    ch_sparsity=0.4,  # prune 40% channels
    ignored_layers=ignored_layers,
)

# Bắt đầu prune
pruner.step()

# Save model
torch.save(model, '100e203_pruned.pt')
