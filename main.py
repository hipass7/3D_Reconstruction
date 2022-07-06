import cv2 # opencv 라이브러리
import numpy as np
import os
import glob # 주어진 경로에 저장된 이미지들을 추출하기 위한 라이브러리
import open3d  # 맵을 포인트 클라우드로 나타내기 위한 라이브러리

#################################
# 카메라 캘리브레이션 (참조 : https://learnopencv.com/camera-calibration-using-opencv/)
#################################

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 26, 0.001)

objpoints = []
imgpoints = [] 

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

images = glob.glob('./calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,
                                             CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img)
    cv2.waitKey()
    
cv2.destroyAllWindows()

h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
k_undist, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print("Camera matrix : \n")
print(mtx)

print("dist : \n")
print(dist)

print("undist : \n")
print(k_undist)

#################################
# 왜곡된 이미지를 undistort 해주기 (참조 : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
#################################

stereo_left = cv2.imread('left.jpg')
h, w = stereo_left.shape[:2]

dst = cv2.undistort(stereo_left, mtx, dist, None, k_undist)
cv2.imwrite('stereo_left.jpg',dst)

stereo_right = cv2.imread('right.jpg')
dst = cv2.undistort(stereo_right, mtx, dist, None, k_undist)
cv2.imwrite('stereo_right.jpg',dst)

#################################
# SIFT 특징점 추출을 이용한 feature matching (참조 : http://www.gisdeveloper.co.kr/?p=6824)
#################################

img1 = cv2.imread('stereo_left.jpg')
img2 = cv2.imread('stereo_right.jpg')
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
img4 = cv2.drawKeypoints(img1, kp1, None)

cv2.imwrite('sift_keypoints.jpg',img3)
cv2.imwrite('feature_extraction.jpg',img4)

#################################
# 스테레오 이미지로부터 depth map 생성 (참조 : https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
#################################

gray_left = cv2.imread('stereo_left.jpg', cv2.IMREAD_GRAYSCALE)
gray_right = cv2.imread('stereo_right.jpg', cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=33)
disparity = stereo.compute(gray_left, gray_right)

disparity = np.uint8(disparity)
disparity_max = np.max(disparity)
disparity_min = np.min(disparity)

for a in np.arange(h):
    for b in np.arange(w):
        if (disparity[a][b] <= disparity_min  or disparity[a][b] >= disparity_max ) :
            disparity[a][b] = 0

depth = disparity.copy()
for i in np.arange(h):
    for j in np.arange(w):
        depth[i][j] = 255 - disparity[i][j]

cv2.imwrite('disparity.jpg', disparity)
cv2.imwrite('depth.jpg', depth)

#################################
# depth map을 이용하여 3D 포인트 클라우드 맵 생성 및 시각화 (참조 : https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0)
#################################

point_cloud = open3d.geometry.PointCloud()

fx, fy, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]

point_cloud_p = []
for i in range(h):
    for j in range(w):
        # if depth[u][v] == 255:
        #     continue
        x = (j - cx) * depth[i][j] / fx
        y = (i - cy) * depth[i][j] / fy
        z = depth[i][j]
        point_cloud_p.append((x, y, z))

point_cloud_p = np.array(point_cloud_p)
filtering = point_cloud_p[:, 2] < 255
points = point_cloud_p[filtering]

point_cloud.points = open3d.utility.Vector3dVector(points)
open3d.visualization.draw_geometries([point_cloud],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)

img = cv2.imread('stereo_left.jpg')
color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
color = np.array(color / 255, dtype=np.float32)
color = color.reshape(h * w, 3)
colors = color[filtering]

point_cloud.colors = open3d.utility.Vector3dVector(colors)
open3d.visualization.draw_geometries([point_cloud],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)

cv2.waitKey()
cv2.destroyAllWindows()

# 자세한 사항은 Report를 참고해주시기 바랍니다.