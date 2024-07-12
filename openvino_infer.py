import time
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from openvino import Core
from ultralytics import YOLO

cls_dict = {0: '未分层',
            1: '分层'}


class OpenvinoInfer:
    def __init__(self, model_path=f'../weights/dissolution_cls_n_openvino_model/dissolution_cls_n.xml', device='GPU'):
        core = Core()
        # 载入并编译模型
        net = core.compile_model(model_path, device_name=device)
        # 获得模型输入输出节点
        self.input_node = net.inputs[0]  # yolov8n-cls只有一个输入节点
        N, C, self.H, self.W = self.input_node.shape  # 获得输入张量的形状
        self.output_node = net.outputs[0]  # yolov8n-cls只有一个输出节点
        self.model = net.create_infer_request()

    # 定义预处理函数
    def preprocess(self, image):
        # Preprocess image data from OpenCV
        [height, width, _] = image.shape
        length = max((height, width))
        letter_box = np.zeros((length, length, 3), np.uint8)
        letter_box[0:height, 0:width] = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(self.W, self.H), swapRB=True)
        return blob

    def infer(self, image):
        blob = self.preprocess(image)
        # 执行推理计算并获得结果
        outs = self.model.infer(blob)[self.output_node]
        # 对推理结果进行后处理
        score = np.max(outs)
        id = np.argmax(outs)
        return score, id

    def infer_det(self, image):
        blob = self.preprocess(image)
        # 执行推理计算并获得结果
        outs = self.model.infer(blob)[self.output_node]
        # 对推理结果进行后处理
        outs.squeeze()
        outs = outs.transpose()
        box = py_cpu_nms(outs, 0.2)
        return box


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


class YoloCls:
    def __init__(self, model_path=None):
        # Load a model
        self.model = YOLO(model_path)

    def train(self, dataset_path, epoch=100, batch=16, **kwargs):
        # Train the model
        results = self.model.train(data=dataset_path, epochs=epoch, batch=batch, imgsz=640, **kwargs)

    def val(self):
        # Validate the model
        metrics = self.model.val()  # no arguments needed, dataset and settings remembered
        top1 = metrics.top1  # top1 accuracy
        top5 = metrics.top5  # top5 accuracy
        return top1, top5

    def predict(self, img_path, **kwargs):
        # Predict with the model
        results = self.model.predict(img_path, **kwargs)  # predict on an image
        probs = results[0].probs.data.cpu().numpy()
        cls = probs.argmax()
        conf = probs[cls]
        return conf, cls


def show(img, cls, conf, i):
    # 写汉字
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('../font/苹方黑体.ttf', 50)
    draw = ImageDraw.Draw(image)
    txt = f'帧  数：{i:04d}\n类  别：' + cls_dict[cls] + '\n' + '置信度：' + str(round(conf, 3))
    draw.text(xy=(10, 20), text=txt, fill=(67, 110, 255), font=font)
    img_show = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img_show


def openvino_infer(model='yolov8n'):
    vino = OpenvinoInfer(f'./weights/{model}_openvino_model/{model}.xml')
    t_list = []
    img = cv2.imread('./zidane.jpg')
    img1 = cv2.imread('./bus.jpg')
    boxes = vino.infer(img)
    for i in range(10):
        t0 = time.time()
        conf, cls = vino.infer(img1)
        # boxes = vino.infer_det(img1)
        t1 = time.time()
        t_list.append(t1 - t0)
    print(f"{model}平均耗时：{round(sum(t_list) / 10, 3)}")


if __name__ == '__main__':
    openvino_infer('yolov8l-cls')
