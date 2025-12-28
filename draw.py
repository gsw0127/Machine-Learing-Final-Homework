import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("23281147pbh_Weight/yolov8m_coco2017/results.csv")

plt.figure()
plt.title('COCO2017 Training and Validation Loss')
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("yolov8m_coco_loss.png")
plt.show()
