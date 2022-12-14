# dynamic_detect_model
장면에서 장소를 인식하는데 중요한 영역을 추출합니다.

### Research Question
장면내에서 장기간으로 정적인 공간요소를 인식을 위한 모델은 무엇인가?

### Our approach
- 모델은 장기간으로 정적일 확률을 출력하는 spatial-wise attention을 사용한다.
- 이 attention의 출력과 모델의 출력은 곱해져서, 장소를 대표하는 global descriptor를 반환한다. 
- spatial-wise attention의 출력을 장기간으로 정적인 공간요소일 픽셀단위의 확률로 사용한다.
- 제안모델은 모델의 contrastive loss를 사용한다.
- 이 확률이 같은 물체에 대해서는 일정할 수 있도록 추가적인 손실함수를 더해야한다. 
- 우리는 물체의 경계지점에서 서로다른 확률을 가질 수 있도록 손실함수를 추가하였다.

### train 방법

train.sh 파일 내부를 수정하고, 아래의 명령어를 입력해주세요. 수정할 파라미터는 train.py에 설명되어있습니다.
```
bash train.sh
```

### test 방법

test.py를 실행하고, 적절한 인자를 넣어주세요.
```
python3 test.py --model_path=$model path$ --dataset_path=$image path$ --image_height=360 --image_width=1280
```

### Dataset 구성 [1]
- train : Kitti 데이터셋[2]의 05 sequence
- validation : Kitti 데이터셋의 06 sequence

### Loss 구성
- loss1 : contrastive loss
- loss2 : (임시) spatial 출력에 대한 loss

### 진행사항
- 22/12/05 : train, validation, model, loss, weight-spatial save 구현


[1] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving?
The KITTI vision benchmark suite,” in Proc. IEEE Conf. Comput. Vis.
Pattern Recognit., 2012, pp. 3354–3361.

[2] H. Osman, N. Darwish and A. Bayoumi, "LoopNet: Where to Focus? Detecting Loop Closures in Dynamic
Scenes," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 2031-2038, April 2022