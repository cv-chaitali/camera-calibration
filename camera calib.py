
import os
import numpy as np

import cv2 as cv

CHESS_BOARD_DIM = (8, 6)


SQUARE_SIZE = 24  


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


calib_data_path = "/Users/chaitalibhattacharyya/Desktop/finaltesting/final_calib_data_aver"
CHECK_DIR = os.path.isdir(calib_data_path)


if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')


obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
    -1, 2
)
obj_3D *= SQUARE_SIZE
print(obj_3D)


obj_points_3D = []  
img_points_2D = []  
import glob
image_files = glob.glob("/Users/chaitalibhattacharyya/Desktop/finaltesting/imgs_home/*.jpg")

for image in image_files:
    print(image)
    
    img = cv.imread(image)
    image = cv.cvtColor(img, cv.COLOR_RGB2BGR)

     
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret == True:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_3D, img_points_2D, gray.shape[::-1], None, None)
print("calibrated")

print("duming the data into one files using numpy ")
np.savez(
    f"{calib_data_path}/calibration_matrix_finalAver",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print("loading data stored using numpy savez function\n \n \n")


img = cv.imread('/Users/chaitalibhattacharyya/Desktop/finaltesting/imgs_home/Photo on 2023-01-19 at 12.37.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
mean_error = 0
for i in range(len(obj_points_3D)):
    imgpoints2, _ = cv.projectPoints(obj_points_3D[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(img_points_2D[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(obj_points_3D)) )










