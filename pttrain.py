from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        amp=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        warmup_epochs=5,
        name="yolov8m_visdrone_pretrained",
        cache=True,
        close_mosaic=20,
    )


if __name__ == "__main__":
    main()
