# YOLOv5 Object Detection

Object Detection



## YOLOv5 PyTorch Hub Tutorial

```bash
$ pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = '/home/gahyeon/github/MDE-Object-Detection-Fusion/YoLo_v5/img.jpg'

# Inference
results = model(imgs)

# Results
results.show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0] # img1 predictions (pandas)
```



## 결과

> * Input img
>
> <img src="./img.jpg" alt="img" style="zoom:80%;" />
>
> ```python
> print(results.pandas().xyxy[0])
> ```
>
> ```
>          xmin        ymin        xmax        ymax  confidence  class           name
> 0   23.272074  184.936234   47.713268  230.997498    0.786491     75           vase
> 1  416.257080  217.734085  599.026184  399.042877    0.653514     57          couch
> 2   26.664982  129.269028   64.631042  226.147369    0.611884     58   potted plant
> 3  268.873993  269.005646  427.174896  387.100983    0.295350     60   dining table
> ```
>
> 
>
> * Output img
>
> <img src="/home/gahyeon/github/MDE-Object-Detection-Fusion/YoLo_v5/out_img.PNG" alt="out_img" style="zoom:80%;" />
>
> 



