import sys
import os
# 获取当前脚本所在目录的父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将父目录添加到系统路径中
sys.path.append(parent_dir)
from config import settings
from config import Point
import cv2
import numpy as np
from v8GetTabletennis.coco_utils import COCO_test_helper
import datetime


QUANTIZE_ON = True
 
 
OBJ_THRESH = 0.50
NMS_THRESH = 0.50

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("ball")

coco_id_list = [1]

landingPoint = []   #创建一个存放所有落点中心点的列表

centerPoint = []   #创建一个存放所有中心点的列表

#用于过滤掉置信度低于阈值的检测框
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos] 

    return boxes, classes, scores

#用于去除冗余的检测框，保留最有可能的检测框
def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

# 损失函数
def dfl(position):
    # Distribution Focal Loss (DFL) using NumPy
    x = np.array(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))  # for numerical stability
    y /= y.sum(axis=2, keepdims=True)  # softmax along the specified dimension
    acc_matrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
    y = (y * acc_matrix).sum(axis=2)
    return y

#用于处理预测的位置信息，并将其转换为边界框的坐标
def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

#用于处理模型的输出，并将其转换为边界框、类别置信度和得分
def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def getSlope(a:list):
    '''
    判断是否发生斜率变化
    '''
    flag = False
    
    while len(a)<4:
        a.append(a[-1])

    
    x1,y1=a[0][0],a[0][1]
    x2,y2=a[1][0],a[1][1]
    x3,y3=a[2][0],a[2][1]
    x4,y4=a[3][0],a[3][1]
    # print("值",x1,y1,x2,y2,x3,y3,x4,y4)
    
    #计算前两个点的斜率，检测分母是否为零
    if x2 != x1:
        slope_first_two = (y2 - y1) / (x2 - x1) 
    else:
        slope_first_two = None
        
    # 计算后两个点的斜率，检查分母是否为零
    if x4 != x3:
        slope_last_two = (y4 - y3) / (x4 - x3)
    else:
        slope_last_two = None


    # 检查斜率是否改变

    if slope_first_two is not None and slope_last_two is not None and slope_first_two > 0 and slope_last_two < 0:   #四帧中中间两帧的平均落点，判断该落点是否在桌面上，以及是在哪边的桌面？
        if slope_first_two * slope_last_two < 0:
            
            # landingCenter = ((x2 + x3) / 2, (y2 + y3) / 2)
            
            # b1和b2是两条直线的截距
            b1 = y1 - slope_first_two * x1
            b2 = y3 - slope_last_two * x3
            X = int((b2 - b1)/(slope_first_two - slope_last_two))
            Y = int(slope_first_two * X + b1)
            landingCenter = (X, Y)
            
            print('slope_first_two：', slope_first_two)
            print('slope_last_two：', slope_last_two)
            print("四个点的坐标是：",a)
            print('落点坐标是：',landingCenter)
            flag = True
            return landingCenter, flag

    return None, False

# 球桌角点区域
def readCorner(path):
    fr = open(path)
    numberOfLines = len(fr.readlines())
    fr.seek(0, 0)
    pointMat = np.zeros((numberOfLines, 2))
    # 重新定位光标
    fr.seek(0, 0)
    index = 0
    for line in fr.readlines():
        # 删除头尾的空格和换行符
        line = line.strip()
        listFormLine = line.split(',')
        pointMat[index, :] = listFormLine[0:2]
        settings.get_value('XCorner', -1).append(int(listFormLine[0]))
        settings.get_value('YCorner', -1).append(int(listFormLine[1]))
        index += 1
    print("############################感兴趣区域角点坐标加载成功")

def getLandingArea(landingCenter):
    '''
    根据落点的坐标判断在哪个区域
    '''
    region=[0 for _ in range(9)]
    regionArea=None

def draw(image, boxes, scores, classes):
    # fourcc=cv2.VideoWriter_fourcc('M','J','P','G')  # 使用mp4编解码器  
    # out = cv2.VideoWriter('output.avi', fourcc, 60.0, (image.shape[1], image.shape[0]))
    # fourcc=cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编解码器  
    # out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (image.shape[1], image.shape[0]))
    # print((image.shape[1], image.shape[0]))
    flag = False
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        
        #bottom, left, right, top = [int(_b) for _b in box]
        center_x=(top+right)//2    #使用整除得到整数横坐标
        center_y=(left+bottom)//2    #使用整除得到整数纵坐标
        
        cc_time=datetime.datetime.now()
        ff_time=cc_time.strftime("%Y-%m-%d %H:%M:%S")
        tt_points_file='tt_points.txt'
        with open(tt_points_file,'a',encoding='utf=8') as file:
            file.write(str((center_x,center_y))+str(ff_time)+str(settings.get_value("tt_point",-1))+'\n')
        centerPoint.append((center_x,center_y))
        landingPoint.extend(centerPoint[-4:])
        

        
        
        # landingPoint.extend([(1,1),(2,2),(3,3)])
        
        # getSlope(landingPoint)  #如果这个地方为真，则发生了斜率的改变返回四帧，否则没有到落点
        # print("四帧计算得到的关键帧的乒乓球位置",getSlope(landingPoint))
        # print('四帧图像的乒乓球位置',centerPoint[-4:])
        # settings.set_value("tt_point",getSlope(centerPoint[-4:]),-1)
        point,flag = getSlope(centerPoint[-4:])
        settings.set_value("tt_point",point,-1)
        
        if flag:
            centerPoint.clear()
            
        # print(settings.get_value("tt_point",-1))
        
        getLandingPoints(settings.get_value("tt_point",-1))     #获取关键帧判断的乒乓球落点
        
        
        
        
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        #cv2.rectangle(image, (bottom, left), (right, top), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # out.write(image) 
    # 释放视频写入器  
    # out.release() 

def getLandingPoints(tt_points):
    
    if tt_points is None:
        return
    
    tt_area=settings.get_value("tt_area",-1)    #[(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
    tt_area_nums=settings.get_value("tt_area_nums",-1)
    # print(tt_area_nums) #{'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0}
    
    for i, area in enumerate(tt_area):
        if point_in_polygon(tt_points, area):
            letter = chr(65 + i)  # Convert index to corresponding letter A-I.
            tt_area_nums[letter] += 1
            break
    
    
    settings.set_value('tt_area_nums',tt_area_nums,-1)
    print(settings.get_value('tt_area_nums',-1))
    
    c_time=datetime.datetime.now()
    f_time=c_time.strftime("%Y-%m-%d %H:%M:%S")
    txt_file='tt_landing_points.txt'
    with open(txt_file,'a',encoding='utf=8') as file:
        file.write(str(settings.get_value('tt_area_nums',-1))+str(f_time)+str(settings.get_value("tt_point",-1))+'\n')

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using the ray casting algorithm.
       使用射线投射算法检查一个点是否位于多边形内部。
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1,n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def readIMG(model,img_src,draw_flag):
    
    # if img_src is None:
    #     print("文件不存在\n")
    
    co_helper = COCO_test_helper(enable_letter_box=True)
    pad_color = (0,0,0)
    img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = np.transpose(img, (2, 0, 1))
    img2 = np.expand_dims(img, axis=0)
    # print("img shape")
    # print(img2.shape)
    
    # print('--> Running model')
    outputs = model.inference(inputs=[img2])

    boxes, classes, scores = post_process(outputs)
    # print(boxes, classes, scores)
    if scores is None:
        return img_src,[],[],[]
    if draw_flag==True:
        draw(img_src, co_helper.get_real_box(boxes), scores, classes)

    return img_src, boxes, classes, scores

# 画出乒乓球桌区域
def draw_polygons_on_frame(frame, polygons):
    if(len(polygons) < 9):
        color = (0,0,255)
    else:
        color = (0,255,0)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

def sharpen_image(image):
    """
    对单个图像帧进行锐化处理。

    :param image: 输入图像帧 (NumPy 数组) 
    :return: 锐化后的图像帧
    """
    # 定义一个简单的锐化内核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # 应用卷积操作
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
      
def myFunc(rknn_lite, IMG):
    
    img_p, boxes, classes, scores = readIMG(rknn_lite,IMG,True)
    if scores!=[]:
        zipped=zip(boxes, scores, classes)
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        
        # 得到最高得分的值
        highest = max(zipped, key=lambda x: x[1])
    # if boxes is not None:
    #     # print("begin drawing")
    #     draw(IMG, boxes, scores, classes) 
    return img_p



