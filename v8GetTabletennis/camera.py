import cv2
import time
import os

# 创建保存视频的目录（如果不存在）
save_dir = 'video'
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头 (默认是 0，表示第一个摄像头设备)
cap = cv2.VideoCapture(0)

# 设置视频流格式和分辨率
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置合理的帧率

# 设置输出视频文件名和编码器0
filename = os.path.join(save_dir, 'test_01_720_01.avi')
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=30, frameSize=(1280, 720))

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 开始录制时间
stime = time.time()

while True:
    # 读取摄像头的一帧图像
    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头图像")
        break

    # 写入视频文件
    out.write(frame)

    # 显示图像（可选）
    cv2.imshow('Camera Frame', frame)

    # 控制录制时间
    if time.time() - stime > 35:
        print("录制时间到，停止录制")
        break

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
out.release()
cv2.destroyAllWindows()