import torch
import numpy as np
from model import LDRN
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
import argparse
import os
import time
import cv2
from y_utils import load_model, score_frame, class_to_label, plot_boxes,depth_plot_boxes

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# Dataloader setting
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--pretrained', type=str, default = "NYU")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--n_Group', type=int, default = 32)
parser.add_argument('--reduction', type=int, default = 16)
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# GPU setting
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--rank', type=int,   help='node rank for distributed training', default=0)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= '0';
cudnn.benchmark = True



model_dir = './pretrained/LDRN_NYU_ResNext101_pretrained_data.pkl'
img_dir = './example/class.jpg'

# MDE Model
max_depth = 10.0
Model = LDRN(args)
Model = Model.cuda()
Model = torch.nn.DataParallel(Model)

Model.load_state_dict(torch.load(model_dir))
Model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# YOLO Model
yolo_model = load_model()

prevTime = 0


    # img = Image.open(img_dir)
a = cv2.imread("/home/junha/MDE-Object-Detection-Fusion/Depth_Yolo_Fusion/rgb_image.jpg")
a = cv2.resize(a,(432,432))
img = a
img = np.asarray(img, dtype=np.float32)/255.0
if img.ndim == 2:
    img = np.expand_dims(img,2)
    img = np.repeat(img,3,2)
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).float()
img = normalize(img)
if args.cuda and torch.cuda.is_available():
    img = img.cuda()

_, org_h, org_w = img.shape
img = img.unsqueeze(0)
new_h = 432 #432
new_w = org_w * (new_h/org_h)
new_w = int((new_w//16)*16)
img = F.interpolate(img, (new_h, new_w), mode='bilinear')

img_flip = torch.flip(img,[3])
with torch.no_grad():
    _, out = Model(img)
    _, out_flip = Model(img_flip)
    out_flip = torch.flip(out_flip,[3])
    out = 0.5*(out + out_flip)
    
if new_h > org_h:
    out = F.interpolate(out, (org_h, org_w), mode='bilinear')
out = out[0,0]
out = out*1000.0
out = out.cpu().detach().numpy().astype(np.uint16)
out = (out/out.max())#*255.0


curTime = time.time()
sec = curTime - prevTime
prevTime = curTime
fps = 1/(sec)

print(fps,end='\r')

result = score_frame(yolo_model,a)
yolo_detect = plot_boxes(yolo_model,result,a)
depth_out = depth_plot_boxes(yolo_model,result,out)

cv2.imshow("depth", out)
cv2.imshow("depth_out", depth_out)
cv2.imshow("detection", yolo_detect)
cv2.imshow("camera", a)

cv2.waitKey(0)

