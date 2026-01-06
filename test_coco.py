from ultralytics import YOLO

# 加载模型
model = YOLO("runs/visdrone_detect/yolov8m_coco2017/weights/best.pt")

results = model.val(data="coco.yaml", batch=1, device="cpu", imgsz=640, verbose=False)

cpu_time = sum(results.speed.values())
print(f"CPU Time: {cpu_time:.2f} ms/image")
