import cv2
import time
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc
from func import draw_polygons_on_frame
from func import sharpen_image
import config

import argparse
tt_area = [
    [config.settings._param_dict['rightSide_line_point_start'], config.settings._param_dict['rightSide_line_point_end']],
    [config.settings._param_dict['mid_line_point_start'], config.settings._param_dict['mid_line_point_end']],
    [config.settings._param_dict['fire_line_point_start'], config.settings._param_dict['fire_line_point_end']]
]

    
if __name__ == '__main__':
    # from rknn_executor import RKNN_model_container 
    print(len(tt_area))
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default='/root/temp/tongyao/v8GetTabletennis/model/yolov8nball_zoo.rknn', help='model path, could be .rknn file')
    
    parser.add_argument('--file_mode', type=str, default='v', help='v--video i--image c--camera')
    
    parser.add_argument('--file_path', type=str, default='./datasets/pingpangte.mp4', help='opencv video file path')    
    
    # parser.add_argument('--file_path', type=str, default='./datasets/a0004.jpg', help='opencv video file path')
    
    parser.add_argument('--img_show', default=False, help='draw the result and show')
    parser.add_argument('--img_save', default=True, help='save the result')
    parser.add_argument('--img_s_path',  type=str, default='./image_s',help='the img save file' )
    parser.add_argument('--video_show', default=False, help='draw the result and show')
    parser.add_argument('--video_save', default=False, help='draw the result and show')
    parser.add_argument('--video_s_path', type=str, default='./video_s',help='draw the result and show')
    # 1280X720   60fps   MJPG
    # 800X600   60fps   MJPG
    # 640X480   120fps   MJPG
    parser.add_argument('--camera_fps', type=str, default='60',help='the camera fps')
    parser.add_argument('--camera_width', type=str, default='1280',help='the camera width')
    parser.add_argument('--camera_height', type=str, default='720',help='the camera height')
   
    args = parser.parse_args()
    # model = RKNN_model_container(args.model_path) 
    
    # readFile(model,args)
    
    # cap = cv2.VideoCapture(args.file_path)
    # cap = cv2.VideoCapture('/root/temp/tongyao/v8GetTabletennis/video/test_02.mp4')
    # cap = cv2.VideoCapture('/root/temp/tongyao/v8GetTabletennis/video/test_720_60_02.avi')
    cap = cv2.VideoCapture(0)
    width = 1280  # 定义摄像头获取图像宽度
    height = 720  # 定义摄像头获取图像长度
    fps=60
    playback_speed = 1
    cap.set(6, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 视频流格式 
    cap.set(5, fps)  # 帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度
    
    # 线程数, 增大可提高帧率
    TPEs = 6
    # 初始化rknn池
    pool = rknnPoolExecutor(
        rknnModel=args.model_path,
        TPEs=TPEs,
        func=myFunc)

    # 初始化异步所需要的帧
    if (cap.isOpened()):
        for i in range(TPEs + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(frame)

    frames, loopTime, initTime = 0, time.time(), time.time()
    while (cap.isOpened()):
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        # frame = sharpen_image(frame)
        draw_polygons_on_frame(frame, config.settings._param_dict['tt_area'])
        draw_polygons_on_frame(frame, tt_area)
        pool.put(frame)
        frame, flag = pool.get()    #这是画完框的图 
        if flag == False:
            break
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            # print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
            loopTime = time.time()
            

    print("总平均帧率\t", frames / (time.time() - initTime))
    # 释放cap和rknn线程池
    cap.release()
    cv2.destroyAllWindows()
    pool.release()
    
    
    
