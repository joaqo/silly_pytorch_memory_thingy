# MEMORY and WEIGHTS Experiments with: https://pytorch.org/vision/stable/models.html

import torch
import torchvision.models as models

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_all_buffers(model):
    return sum(p.numel() for p in model.buffers())

def print_counts(model, name=""):
    print(name)
    print("Trainable Parameters:", count_trainable_parameters(model) / 1_000_000, "million")
    print("Parameters:", count_all_parameters(model) / 1_000_000, "million")
    print("Buffers:", count_all_buffers(model) / 1_000_000, "million\n")

def print_buffer_data(model):
    """It's basically all due to batchnorm, lol"""
    print("Buffers")
    for name, buff in model.named_buffers():
        print(name, buff.numel())
    print()

def print_param_data(model):
    print("Parameters")
    for name, param in model.named_parameters():
        print(name, param.numel())
    print()

print_counts(models.resnet18(), "resnet18")
print_counts(models.resnet50(), "resnet50")
print_counts(models.resnet101(), "resnet101")
print_counts(models.mobilenet_v3_small(), "mobilenet_v3_small")
print_counts(models.detection.fasterrcnn_resnet50_fpn(), "fasterrcnn_resnet50_fpn")
print_counts(models.detection.maskrcnn_resnet50_fpn(), "maskrcnn_resnet50_fpn")

print_buffer_data(models.detection.maskrcnn_resnet50_fpn())
print_param_data(models.detection.maskrcnn_resnet50_fpn())
print("BatchNorm")
print_param_data(torch.nn.BatchNorm2d(3))

# Memory tests. Run while looking at `watch -n 0.5 nvidia-smi` and
# then compare becuse there are around 1284MB of runtime VRAM overhead.
random_input = torch.rand(1, 3, 224, 224).cuda()

# Load CUDA runtime on GPU or something, 1284MB of VRAM. TODO: Find out what this is
resnet = models.resnet50(pretrained=True).cuda()
input("Look at current memory and then compare")

mode = "forward_no_grad"

if mode == "forward_backward_grad":
    # Forward and backward with grad, 344MB of VRAM
    labels = torch.tensor([1], dtype=torch.long).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    output = resnet(random_input)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
elif mode == "forward_grad":
    # Forward with grad, 248MB of VRAM
    resnet(random_input)
elif mode == "forward_no_grad":
    resnet.eval()  # <- Makes no difference in VRAM
    # Forwward with no grad, 174MB of VRAM
    with torch.no_grad():
        resnet(random_input)
else:
    print("LOL WHAT?")

input("Look at memory")
