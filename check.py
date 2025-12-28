import torch

print("\n===== PyTorch CUDA 检测 =====")

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n--- CUDA 设备 {i} ---")
        print("名称:", torch.cuda.get_device_name(i))
        print("显存总量:", round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2), "GB")
else:
    print("\n PyTorch 无法检测到 GPU，请检查 CUDA / PyTorch 安装是否正确。")

print("\n===== 建议在 YOLO 中设置的 device 参数 =====")
if torch.cuda.is_available():
    print("YOLO 训练时应该写: device='0'")
else:
    print("当前只能使用 CPU: device='cpu'")
