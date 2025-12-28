from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="coco.yaml",
        epochs=300,
        imgsz=640,
        pretrained=True,
        amp=True,
        device=0,
        batch=32,
        workers=8,
        name="yolov8m_coco2017"
    )

if __name__ == '__main__':
    main()