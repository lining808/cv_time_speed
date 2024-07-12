import os
import time
import cv2
from ultralytics import YOLO

from openvino_infer import openvino_infer


def infer(model_name='yolov8s'):

    model_path = f'./weights/{model_name}.pt'
    img = cv2.imread('./zidane.jpg')
    img1 = cv2.imread('./bus.jpg')
    model = YOLO(model_path)
    res = model.predict(img)
    t_list = []
    for i in range(10):
        t0 = time.time()
        res1 = model.predict(img1)
        t1 = time.time()
        t = t1 - t0
        t_list.append(t)
    print(f'{model_name}推理耗时：', round(sum(t_list) / 10, 3))


def tensorrt_infer(model_name='yolov8s'):

    model_path = f'./weights/{model_name}.engine'
    img = cv2.imread('./zidane.jpg')
    img1 = cv2.imread('./bus.jpg')
    model = YOLO(model_path)
    res = model.predict(img)
    t_list = []
    for i in range(10):
        t0 = time.time()
        res1 = model.predict(img1)
        t1 = time.time()
        t = t1 - t0
        t_list.append(t)
    print(f'{model_name}推理耗时：', round(sum(t_list) / 10, 3))

def onnx_infer(model_name='yolov8s'):

    model_path = f'./weights/{model_name}.onnx'
    img = cv2.imread('./zidane.jpg')
    img1 = cv2.imread('./bus.jpg')
    model = YOLO(model_path)
    res = model.predict(img)
    t_list = []
    for i in range(10):
        t0 = time.time()
        res1 = model.predict(img1)
        t1 = time.time()
        t = t1 - t0
        t_list.append(t)
    print(f'{model_name}推理耗时：', round(sum(t_list) / 10, 3))

def export_openvino(weights_path='./weights/'):
    for file in os.listdir(weights_path):
        if 'obb' in file:
            model = YOLO(weights_path+file)
            model.export(format='openvino')


def export_engine(weights_path='./weights/'):
    for file in os.listdir(weights_path):
        model = YOLO(weights_path+file)
        model.export(format='engine')


if __name__ == '__main__':
    # infer('yolov8l-obb')
    onnx_infer('yolov8s')
    # openvino_infer('yolov8l-obb')
    # tensorrt_infer('yolov8s')
    # export_openvino('./weights/')
