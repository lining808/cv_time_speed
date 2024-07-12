import time
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet50 = models.mobilenet_v3_large(pretrained=True)
resnet50.eval().to(device)

img_rgb = Image.open('./bus.jpg').convert('RGB')
norm_mean = [0.485, 0.456, 0.406]    # 均值
norm_std = [0.229, 0.224, 0.225]     # 方差
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),  # 转换到-1 - 1
])
t_list = []
for i in range(10):
    t0 = time.time()
    img_tensor = inference_transform(img_rgb)
    img_tensor.unsqueeze_(0)    # 1x3x800x800
    img_tensor = img_tensor.to(device)   # use cuda

    with torch.no_grad():
        output = torch.nn.functional.softmax(resnet50(img_tensor), dim=1)

    t1 = time.time()
    t_list.append(t1-t0)
print('推理耗时：', round(sum(t_list)/10, 3))