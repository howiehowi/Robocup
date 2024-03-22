from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

#rbbox带角度检测
model=YOLO('D:/PycharmProjects/robocup_getCoordinate/robocup_obb.pt')
results=model('D:/PycharmProjects/robocup_getCoordinate/rgbImg.png',show=True,save=False)

centers = []
angles = []
for r in results:
    # 获取边界框的xywhr表示，其中x和y代表中心点坐标
    boxes = r.obb.xywhr
    # 将每个边界框的中心点添加到centers列表中，并转换为CPU上的普通数字
    for box in boxes:
        center_x, center_y = box[0].item(), box[1].item()
        centers.append((center_x, center_y))

        # 将弧度转化为度数
        angle_in_degrees = box[4].item() * (180 / torch.pi)
        angles.append(angle_in_degrees)
        print(f"Angle in degrees: {angle_in_degrees}")

# 打印所有的中心点坐标
print(centers)

img = cv2.imread('D:/PycharmProjects/robocup_getCoordinate/rgbImg.png')

# 对于每个检测到的物体，绘制中心点并显示(x, y, angle)
for center, angle in zip(centers, angles):
    x, y = int(center[0]), int(center[1])
    text = f"({x}, {y}, {int(angle)})"  # 格式化字符串，包括坐标和角度，精确到小数点后两位

    # 绘制中心点
    cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # 在中心点上方显示文本
    cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# 将BGR图像转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
plt.imshow(img_rgb)
plt.axis('off')  # 不显示坐标轴
plt.show()