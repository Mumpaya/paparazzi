# slightly modified from
# https://gist.github.com/mesutpiskin/0ced27981487491403610324fea55038
import warnings

import numpy as np
import cv2
import glob
import os

def run_calibration():
    def load_image(path):
        img = cv2.imread(path)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    # path to images
    folder_path = r"C:\Users\super\Downloads\AE4317_2019_datasets\AE4317_2019_datasets\calibration_frontcam\20190121-163447"
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    good_im = np.zeros(len(image_files), dtype=bool)
    if len(image_files) == 0:
        raise Exception("No images found in the specified folder. Please check the path")

    # Define the chess board rows and columns
    CHECKERBOARD = (6,9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    counter = 0
    i = 0
    for path in image_files[::]:
        # Load the image and convert it to gray scale
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # Make sure the chess board pattern was found in the image
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            #cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
            good_im[i] = True
            counter+=1
        print(str(path))
        i +=1

    print(f'Found {counter}/{i} good images for calibration')
    N_imm = counter# number of calibration images
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (520, 240), np.eye(3),
                                                                balance=0.0, new_size=(520, 240))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, (520, 240), cv2.CV_16SC2)

    for_save = {
        'K': K,
        'D': D,
        'Knew': Knew,
    }
    np.save('calibration_data.npy', for_save)

    for i in range(len(image_files)):
        img = load_image(image_files[i])
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow('Original Image', img)
        cv2.imshow('Undistort Image', undistorted_img)
        cv2.waitKey(100)


K = np.array([
    [324.5960989 ,   0.        , 265.97140012],
     [0.         ,325.14620072 ,213.11778828],
    [0.          , 0.          , 1.],
    ])
D = np.array([
    [-0.05242866],
     [0.05816831],
     [-0.10717978],
     [0.06408123],
    ])
Knew = np.array([
    [293.2446961 ,   0.     ,    269.86206627],
 [  0.     ,    293.74166585, 231.41389943],
 [  0.    ,       0.   ,        1.        ],
])

if os.path.exists('calibration_data.npy'):
    correction_data = np.load('calibration_data.npy', allow_pickle=True).item()
    K = correction_data['K']
    D = correction_data['D']
    Knew = correction_data['Knew']
else:
    warnings.warn('Calibration data not found, using hardcoded values. Run run_calibration() to generate calibration data.')

fx, fy = Knew[0, 0], Knew[1, 1]
cx, cy = Knew[0, 2], Knew[1, 2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, (520, 240), cv2.CV_16SC2)


def undistort_image(image):
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def load_image(path):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = undistort_image(img)
    return img

if __name__ == "__main__":
    run_calibration()