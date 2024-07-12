# 常用视觉分类、目标检测模型性能测试

测试常用模型在单张图像上的识别速度，不包含图像读取时间，但包含图像预处理。可以在以后的应用中根据硬件配置选取合适的模型，达到最佳效果。其中推理速度为正常推理的速度，加速CPU使用openvino加速，GPU使用tensorrt加速。
CPU硬件： Intel i7 11700 16GB
GPU硬件： Nvidia rtx 3090 24GB


## CPU

硬件： i7 11700 16GB
目标检测
其中推理速度单位为秒，mAP准确率在COCO数据集得到。旋转目标检测mAP在DOTAv1数据集得到。
| 模型          | 推理速度  | 加速    | mAP@50-95 |
|-------------|-------|-------|-----------|  
| yolov8n     | 0.128 | 0.024 | 37.3      |
| yolov8s     | 0.323 | 0.053 | 44.9      |
| yolov8m     | 0.648 | 0.108 | 50.2      |
| yolov8l     | 1.252 | 0.236 | 52.9      |
| yolov9n     | 0.177 | 0.029 | 38.3      |
| yolov9s     | 0.372 | 0.05  | 46.8      |
| yolov9m     | 0.886 | 0.115 | 51.4      |
| yolov9l     | 1.239 | 0.148 | 53.0      |
| yolov10n    | 0.172 | 0.043 | 38.5      |
| yolov10s    | 0.365 | 0.075 | 46.3      |
| yolov10m    | 0.818 | 0.138 | 51.1      |
| yolov10l    | 1.374 | 0.242 | 53.2      |
| yolov8n-obb | 0.311 | 0.051 | 78.0      |
| yolov8s-obb | 0.717 | 0.157 | 79.5      |
| yolov8m-obb | 1.635 | 0.279 | 80.5      |
| yolov8l-obb | 3.139 | 1.127 | 80.7      |

图像分类
其中推理速度单位为秒，Top-1准确率在ImageNet数据集得到。
| 模型                 | 推理速度  | 加速    | Top-1 |
|--------------------|-------|-------|-------|  
| yolov8n-cls        | 0.017 | 0.005 | 69.0  |
| yolov8s-cls        | 0.037 | 0.007 | 73.8  |
| yolov8m-cls        | 0.076 | 0.011 | 76.8  |
| yolov8l-cls        | 0.146 | 0.029 | 76.8  |
| yolov8x-cls        | 0.25  |       | 79.0  |
| resnet18           | 0.306 |       | 72.1  |
| resnet34           | 0.418 |       | 75.5  |
| resnet50           | 0.903 |       | 77.2  |
| resnet101          | 1.614 |       | 78.3  |
| mobilenet_v3_small | 0.093 |       | 67.4  |
| mobilenet_v3_large | 0.252 |       | 75.2  |
| efficientnet_v2_s  | 0.988 |       | 83.9  |
| efficientnet_v2_m  | 1.684 |       | 85.1  |
| swin_v2_t          | 1.412 |       | 81.6  |
| swin_v2_b          | 4.074 |       | 84.1  |
| convnext_tiny      | 0.766 |       | 82.9  |
| convnext_base      | 2.363 |       | 85.8  |

## GPU

硬件： 3090 24GB
目标检测

| 模型          | 推理速度  | 加速    | mAP@50-95 |
|-------------|-------|-------|-----------|  
| yolov8n     | 0.025 | 0.007 | 37.3      |
| yolov8s     | 0.023 | 0.008 | 44.9      |
| yolov8m     | 0.026 | 0.011 | 50.2      |
| yolov8l     | 0.026 | 0.015 | 52.9      |
| yolov9n     | 0.033 | 0.008 | 38.3      |
| yolov9s     | 0.032 | 0.008 | 46.8      |
| yolov9m     | 0.038 | 0.012 | 51.4      |
| yolov9l     | 0.026 | 0.013 | 53.0      |
| yolov10n    | 0.018 | 0.006 | 38.5      |
| yolov10s    | 0.019 | 0.007 | 46.3      |
| yolov10m    | 0.025 | 0.009 | 51.1      |
| yolov10l    | 0.024 | 0.013 | 53.2      |
| yolov8n-obb | 0.047 | 0.006 | 78.0      |
| yolov8s-obb | 0.03  | 0.008 | 79.5      |
| yolov8m-obb | 0.039 | 0.014 | 80.5      |
| yolov8l-obb | 0.041 | 0.023 | 80.7      |

图像分类

| 模型                 | 推理速度  | 加速    | Top-1 |
|--------------------|-------|-------|-------|  
| yolov8n-cls        | 0.012 | 0.021 | 69.0  |
| yolov8s-cls        | 0.012 | 0.02  | 73.8  |
| yolov8m-cls        | 0.013 | 0.027 | 76.8  |
| yolov8l-cls        | 0.014 | 0.029 | 76.8  |
| yolov8x-cls        | 0.016 | 0.03  | 79.0  |
| resnet18           | 0.042 |       | 72.1  |
| resnet34           | 0.046 |       | 75.5  |
| resnet50           | 0.055 |       | 77.2  |
| resnet101          | 0.063 |       | 78.3  |
| mobilenet_v3_small | 0.054 |       | 67.4  |
| mobilenet_v3_large | 0.056 |       | 75.2  |
| efficientnet_v2_s  | 0.074 |       | 83.9  |
| efficientnet_v2_m  | 0.076 |       | 85.1  |
| swin_v2_t          | 0.127 |       | 81.6  |
| swin_v2_b          | 0.145 |       | 84.1  |
| convnext_tiny      | 0.048 |       | 82.9  |
| convnext_base      | 0.068 |       | 85.8  |

# 结论

总体来说YOLO不论是分类还是目标检测，基本上做到了速度和精度的均衡。
openvino加速可以比CPU推理快6倍左右，但需要CPU是英特尔平台并且有集成显卡。精度有一定程度下降，平均下降2-3%。onnx推理精度几乎保持不变，速度提升约3倍。
tensorrt加速可以比GPU推理快3倍左右，需要GPU为英伟达平台。精度基本保持不变，下降在1%内。

图像分类

|     | 速度          | 均衡          | 精度                |
|-----|-------------|-------------|-------------------|
| CPU | yolov8n-cls | yolov8m-cls | efficientnet_v2_m |
| GPU | yolov8n-cls | yolov8m-cls | convnext_base     |

目标检测

|     | 速度       | 均衡       | 精度       |
|-----|----------|----------|----------|
| CPU | yolov8n  | yolov8m  | yolov9l  |
| GPU | yolov10n | yolov10m | yolov10l |

CPU推理有集显使用openvino，无集显使用onnx。

GPU推理使用tensorrt

