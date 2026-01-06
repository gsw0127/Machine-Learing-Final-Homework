import os


def generate_image_list(image_dir, output_file):
    with open(output_file, "w") as f:
        for image_file in os.listdir(image_dir):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                # 将图片的相对路径写入文件
                f.write(f"{image_dir}/{image_file}\n")


# 定义路径
train_image_dir = "images/train2017"
val_image_dir = "images/val2017"

train_output_file = "train2017.txt"
val_output_file = "val2017.txt"

# 生成train2017.txt
generate_image_list(train_image_dir, train_output_file)

# 生成val2017.txt
generate_image_list(val_image_dir, val_output_file)

print("train2017.txt 和 val2017.txt 生成完成！")
