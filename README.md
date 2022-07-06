# 3D_Reconstruction
2022-1 Robot Sensor Processing Project




로봇센서데이터처리

MID-TERM REPORT

< 3D RECONSTRUCTION >









~ 2022. 05. 22.




응용물리학과 4학년

2017103038 권인회








< 목       차 >




I . 들어가며


II . Camera Calibration


III . Feature Extraction & Matching


IV . Disparity & Depth Map


V . 3D Point Cloud Reconstruction


VI . 마치며















I . 들어가며
1) 머리말
두 장의 이미지를 가지고 3D로 복구하는 것은 이론상으로만 이해를 해왔고 직접 해 본 적은 없었다. 이 수업을 통해 이론을 탄탄히 배우고 중간고사 대체과제를 통해 실습을 해 볼 수 있는 기회를 맞이할 수 있었다. 실습을 하기 위해서는 사용하는 라이브러리(openCV, open3D 등)에 대한 이해가 필요하였기 때문에 그 부분을 위주로 공부하여 적용하였다.

2) 사용법
보고서를 시작하기 전, 첨부한 코드의 사용법을 명시한 뒤, 시작하려고 한다. Camera Calibration과 3D Visualization는 창으로 결과물을 확인할 수 있으며, 그 외에는 모두 파일로 출력이 된다.

- Inputs
1. calibration 폴더 안에 체크보드(9x6) 이미지 6장 이상
2. 기본 폴더 안에 스테레오 이미지 2장

- Outputs
1. 왜곡 보정된 스테레오 이미지 2장 (stereo_left, stereo_right)
2. SIFT 특징점 추출된 이미지 1장 (feature_extraction)
3. 두 이미지에서 SIFT 특징점 매칭된 이미지 1장 (sift_keypoints)
4. disparity와 depth 맵 이미지 2장 (disparity, depth)
5. open3D 창을 통한 3D Visualization 창 (색상 미포함)
6. open3D 창을 통한 3D Visualization 창 (색상 포함)

- 매뉴얼
1. 압축을 풀고 calibration 폴더 안 이미지 7장과 left, right 이미지 2장을 확인한 뒤, main.py를 실행시킨다.
2. 처음 나타나는 open3D 창은 색상이 포함되지 않은 3D Point Cloud Visualization이며, <ESC>를 통해 종료 가능하다.
3. 바로 다음 나타나는 open3D 창은 색상이 포함된 3D Point Cloud Visualization이며, <ESC>를 통해 종료 가능하다.
4. 나머지 Outputs는 해당 폴더에 파일로 생성되므로 확인하면 된다.








II . Camera Calibration
모든 카메라에는 Intrinsic parameter가 존재한다. 처음에는 그 카메라의 제조사 및 모델을 파악하여 검색을 통해 알아내야 하는 것인 줄 알았다. 하지만, 이 Camera Calibration 과정을 통해 파악할 수 있고 이 정보를 알아야 3D 복구 및 왜곡 보정을 진행할 수 있다.



< 그림 1 – Intrinsic matrix >




체크보드에 대한 정보를 설정해주고, 각 체커보드 이미지에 대한 3D 및 2D 점 벡터를 저장할 벡터 리스트를 만들어준다. 이후 3D 점의 세계 좌표를 정의해주면서 이 과정은 라이브러리를 이용한다.



이후 체크보드 이미지를 하나씩 불러오면서 grayscale로 변환해주고, 해당 이미지에서 코너를 찾고 원하는 개수의 코너가 발견되면 true를 반환하고 corner들의 정보를 가져온다. 그 corner들의 픽셀 좌표에 대해서 미세 조정을 하기 위해 cv2.cornerSubPix를 통해 parameter로 주어진 기준 (blocksize 11x11 등)에 따라 미세 조정을 마친 뒤, 그 코너들을 아까 정의해준 리스트에 포함시킨다. 이렇게 알려진 3D 점 벡터 값들과 감지된 코너의 해당 픽셀 좌표를 cv2.calibrateCamera 함수에 parameter로 전달하면 카메라 캘리브레이션이 수행되어 Intrinsic parameter와 렌즈 왜곡 계수가 생성된다.






그리고 생성된 계수들을 cv2.getOptimalNewCameraMatrix 함수에 넣어 왜곡이 보정된 카메라 계수를 얻어낸다. 이 정보를 사용하여 왜곡된 이미지를 보정해줄 것이다.



함수에 맞는 parameter들을 넣고 cv2.undistort 해주면 입력으로 넣은 원본 이미지에서 왜곡을 편 이미지로 바꿔 출력하게 되고, 확인을 할 수 있다.



지금 구한 이 계수들은 다음 단계에 3D reconstruction 과정에서도 이용할 예정이다.

III . Feature Extraction & Matching
다음 과정은 왼쪽 이미지와 오른쪽 이미지가 어느정도 움직였는지에 대해 파악할 필요가 있다. 따라서, 각 이미지에서 SIFT 특징점 추출을 이용하여 특징점을 찾아내고, 각 이미지가 담당하는 특징점끼리 연결하여서 어느정도의 픽셀 차이가 있는지 알아내는 과정이다. 이 정보를 토대로 Disparity를 구하는 데 기여할 것이다.

sift라는 변수를 통해 SIFT 특징점 추출기를 활성화 시켜주고 각 이미지에 대해 특징점과 기술자를 생성해준다. 그리고 cv2.drawKeypoints 함수를 이용하여 이미지에 특징점을 표시하여 나타내준다. 두 이미지의 특징점을 매칭하기 위해서는 cv2.BFMatcher 함수를 이용하면 되고, 거리 측정법에 대해서는 가장 기본적인 유클리드 거리 측정법을 사용하였다. 



카메라를 수평 방향으로만 움직였기 때문에 특징점 매칭을 보면 모두 수평한 직선으로 이루어져있어야 좋은 결과라고 할 수 있지만 간간히 그렇지 않은 직선들이 보이는 것을 확인할 수 있다. 이러한 이유 때문에 3D 복구 과정에서 노이즈가 발생할 수 있다. 그렇지만 해당 과정은 라이브러리가 잘 구성되어 있고 사용법도 쉬워 문제를 잘 해결할 수 있었다.

IV . Disparity & Depth Map


< 그림 2 – disparity 식 >


Disparity는 다음과 같은 공식에 의해 구할 수 있다. Depth를 나타내는 Z값을 알아내기 위해서는 Baseline인 B값, 카메라의 f값, 그리고 disparity값을 알아야한다. 따라서, disparity를 먼저 알아낸 뒤, depth를 구해볼 예정이다. disparity값은 opencv 라이브러리가 기본적으로 제공해주고 있기 때문에 cv2.stereoBM_create 함수를 사용하여 구할 것이다.



여기서 중요한 점은 이미지를 grayscale로 바꾼 뒤 입력을 넣어줘야 한다는 것이다. 그리꼬 또 하나 흥미로운 점은 blockSize를 변경하면서 disparity 이미지가 달라지는 것을 확인할 수 있는데, 이 patch size가 커질수록 더 smooth한 disparity 이미지를 제공해준다. 그렇기 때문에 적당한 blockSize를 설정해주어야하며, 해당 프로젝트에서는 33의 값을 설정해주었다.

이후에는 disparity 값을 depth 값으로 변경해주기 위해 가공해주는 과정이 필요했고 numpy의 부호없는 int 8비트로 변경해주어 255까지의 값을 담을 수 있게 하였다. 이 값을 벗어나는 경우에는 잘못된 disparity라고 판단해야 하기 때문에 픽셀 하나하나를 확인하면서 최소, 최대값 내에 없는 경우 disparity를 0으로 임의로 설정하도록 하였다. 이 과정을 하지 않으면 3D 복구를 했을 때, disparity를 구하지 못한 영역에 대해서 frame이 그대로 남아있기 때문에 outlier 느낌으로 존재한다.

또한, depth 값과 disparity 값을 더하면 흰색(값 255)이 된다는 것을 파악하고 있기 때문에 모든 픽셀에 대해 255에서 disparity 값을 빼주면서 depth 값을 손쉽게 구할 수 있었다. 따라서 다음과 같은 disparity map과 depth map이 나타난다.




왼쪽이 Disparity Map, 오른쪽이 Depth Map인데 가까울수록 Depth값이 낮고 (검정색) 멀수록 Depth값이 높아 (흰색) 잘 만들어진 것을 확인할 수 있다. 물론 흰색 부분에 대해서는 Unknown 영역이라고 생각할 수 있다.

V . 3D Point Cloud Reconstruction

< 그림 3 – Camera Coordinate Projection >


3D 좌표를 2D plane으로 투영했을 때, 좌표가 어떻게 변하는지 다음 행렬로 나타낼 수 있다. 따라서 이 방법을 역이용하여 2D plane의 점을 이용하여 3D 좌표를 얻어낼 수 있다. 이 프로젝트에서는 카메라 내부 파라미터가 fx, fy, cx, cy로 나타나는 것을 감안하여 다음 코드와 같이 식을 나타내어 x, y, z 좌표를 복원할 수 있다.



우선 point cloud를 open3d 라이브러리를 이용하여 선언해주고 이전에 카메라 캘리브레이션을 통해 구했던 카메라의 fx, fy, cx, cy 값을 가져온다. 이후 방금 말했던 2D 좌표를 3D 좌표로 변환하기 위해 depth 값을 이용하여 3D 좌표로 변환한 뒤, point cloud로 나타낼 점 리스트에 추가해준다. 행렬을 생각하면 각 좌표값에 대해 어떻게 식을 세워야하는지 알 수 있다. 이후 필터링 과정을 거쳐야하는데, 추가된 3D 좌표 점들에 대해서 depth 값이 255 이상인 경우 제외를 해준다. 아까와 같은 이유인데, 제외를 해주지 않으면 미추정된 depth 값들이 255값을 가지게 되어 outlier처럼 3D 영상에 나타나게 된다. 따라서 다음과 같이 필터링을 하여 해당되는 index에 있는 값들만 남겨준다.



이후 아까 선언해 준 point cloud에 점 좌표값들을 넣어주고 visualization하면 open3d 창에 다음과 같이 나타난다. 여기서 parameter 값들은 reference를 그대로 이용하였으며 3d 창 내에서 내가 바라보는 시점을 어떻게 할 것인지 정하게 되는 것이다.


다음과 같이 가까울수록 푸른색, 멀수록 붉은색으로 나타나는 것을 확인할 수 있고 중간중간에 노이즈도 있지만 대체적으로 잘 나온 것을 확인할 수 있다. 이제 이 점들에 이미지의 색상을 입히면 3D 복원의 마지막 과정을 맞이할 수 있다.



이미지를 다시 불러와서 opencv에서는 이미지를 BGR 순으로 저장하고 있기 때문에 RGB 순으로 변경해주고 open3D에서는 색상의 값을 0~255가 아닌 0~1의 값으로 사용하고 있기 때문에 255로 나눠준 뒤, size를 다시 맞춰준 뒤 아까 점 리스트에 대해 했던 것처럼 필터링을 거쳐 해당하는 index만 남겨둔다. 이후 점 리스트의 index에 맞게 그 점에 대한 색상이 부여가 되기 때문에 이후에 다시 visualization을 하면 색상까지 올바르게 포함된 것을 확인할 수 있다.

VI . 마치며
1) 느낀점 및 소감
생각보다 만족스러운 결과가 나오지 못했다. reference를 통해서 3D 복원이 된 모습을 봤었는데 그만큼 정확하게 나오지 않은 것 같다. 하지만, 여러 가지 오차가 있을 수 있다는 점을 확인했을 때는 이 정도면 매우 훌륭한 복원이라고 생각한다. 우선, 화면상에서 앞에 있던 키보드와 책이 3D 복원을 했을 때, 앞쪽에 형성되어있고 책장이 뒤쪽에 형성되어 있는 이 정도 구분을 해냈다면 성공적인 것 같다.

하지만 비어있는 공간이 많은 것처럼 이유를 생각해보면, 아마 textureless가 가장 큰 영향을 끼쳤을 것이라고 생각한다. 최대한 texture가 많은 공간에 대해 사진을 찍으려고 하였으나 그래도 이미지 내에 textureless 공간이 없도록 사진을 촬영하는 것은 어려움이 있었고 집 안에서 카메라를 정밀하게 조정할 수 있으면서 찍을 수 있는 사진의 최적이었다. 따라서, 책장 옆의 벽지나 모니터 화면의 경우에는 비어있는 것을 확인할 수 있다.

또한 처음에는 스마트폰으로 촬영을 했었으나 해상도가 매우 높아 계산 시간도 길고, 모니터 화면을 통해서 완전히 확인할 수 없었고, 아무리 카메라를 정밀하게 이동하고 캘리브레이션을 해도 3D 좌표의 복구가 원활하지 않았다. 코드의 문제라고 생각하여 코드를 여러 번 수정하였지만 결과는 같았고, 이에 장비 문제가 있다고 생각하여 웹캠으로 변경하였는데 잘 이루어지는 것을 확인하였다.

하지만, 웹캠을 정밀하게 움직이는 것 또한 어려움이 있었기에 최대한 수평하게 translation하려고 했지만 여기서 약간의 흔들림 등이 존재했을 가능성이 있고, 각도 조절에도 실패하여 책이나 키보드 같은 경우 수직으로 3D 복원이 되도록 하여 더욱 깔끔하게 나타내고 싶었으나, 비스듬히 놓여지는 바람에 조금 더 노이즈가 있는 것처럼 보이지 않는가 싶다.

그리고 또 의아한 점이 카메라가 얼마나 이동했는지, 그리고 체크보드의 한 칸이 몇 mm로 이루어져 있는지 미리 체크를 해두었는데 이를 어디에 사용하는 것인지 확인하지 못 하였다. 처음엔 캘리브레이션을 위한 criteria 값에 넣는 것인줄 알았는데 이 값을 변경하여도 아무런 변화는 없었다. 또한 disparity를 계산하는 과정에서 넣는 parameter에서도 확인할 수 없었다. 아무래도 reference에서 확인했을 때는 Baseline을 포함하여 계산하는 과정이 있었던 것으로 확인했는데, 라이브러리의 요구사항에 포함이 되지는 않는 것 같았다. 그럼에도 좋은 결과가 나왔기에 이렇게 마무리를 짓는다.

2) 참조
1. 그림 1 : (https://learnopencv.com/camera-calibration-using-opencv/)
2. 그림 2 : (https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
3. 그림 3 : (https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf)
4. (https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0)
5. (http://www.gisdeveloper.co.kr/?p=6824)







