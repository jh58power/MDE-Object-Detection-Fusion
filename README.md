---

---

# MDE-Object-Detection-Fusion

Fusion of Monocular Depth Estimation and Object Detection

MDE 알고리즘을 통해 측정된 depth 이미지에 object detection 결과를 fusion 하는 알고리즘 개발



## LapDepth

> * Pretrained Model 로 바로 테스트 가능
>

#### PreTrained Models

* [Trained with NYU Depth V2](https://drive.google.com/file/d/13WyHCmQINyzprCerkOBT_Pf_W-PbWzBi/view?usp=sharing)



./pretrained 폴더 생성 후 모델 넣기

```bash 
$ python3 my_demo.py
```

#### ERROR

> - my_demo.py 코드내 CUDA_VISIBLE_DEVICE 가 자신 PC와 매칭하는지 확인
> - 모델 경로 확인
> - cv2.VideoCapture(num)  -> num 확인 (Ubuntu 환경일 경우 ls /dev/video* 로 확인 가능) 

## Object Detection

* <MDE 관련 테스트 종료 후 추가 예정>





## 진행 현황 

- [ ] Monocular-Depth-Estimation Test
- [ ] Object Detection Test
- [ ] Fusion



### github 관리

* git 에 코드 변경 내용 Push 할시 꼭 Branch 생성 후 자신의 Branch 에 작성하기
* 오류 발생, 진행이 어려운 상황에는 issue 에 해당 문제 작성하기 or Discord 에 작성하기
* 

