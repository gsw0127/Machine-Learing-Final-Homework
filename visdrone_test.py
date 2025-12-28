# from ultralytics import YOLO
# import matplotlib
# matplotlib.use( "TkAgg")
#
# if __name__ == '__main__':
#     # model = YOLO('23281147pbh_Weight/yolov8m_visdrone_scratch/weights/best.pt')  # scratch训练得到的权重文件路径
#     model  = YOLO('23281147pbh_Weight/yolov8m_visdrone_pretrained1/yolov8m_visdrone_pretrained1/weights/best.pt')  # 预训练权重训练结果
#
#     # 直接指定测试集路径
#     metrics = model.val(
#         data='VisDrone.yaml',
#     )
#
#


from ultralytics import YOLO

# 加载模型
model = YOLO("23281147pbh_Weight/yolov8m_visdrone_scratch/weights/best.pt")

results = model.val(data="VisDrone.yaml", split="test", batch=1, device="cpu", imgsz=640, verbose=False)

cpu_time = sum(results.speed.values())
print(f"CPU Time: {cpu_time:.2f} ms/image")
