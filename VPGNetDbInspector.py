import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse


def walkThroughDataset(dataSetDir):
    markClass = [
        ### Lane and road markings (4th channel) ###
        'background',
        'lane_solid_white',
        'lane_broken_white',
        'lane_double_white',
        'lane_solid_yellow',
        'lane_broken_yellow',
        'lane_double_yellow',
        'lane_broken_blue',
        'lane_slow',
        'stop_line',
        'arrow_left',
        'arrow_right',
        'arrow_go_straight',
        'arrow_u_turn',
        'speed_bump',
        'crossWalk',
        'safety_zone',
        'other_road_markings']

    vpClass = [
        ### Vanishing Points (5th channel) ###
        'background',
        'easy',
        'hard',
    ]

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dataSetDir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    colorMapMat = np.zeros((18, 3), dtype=np.uint8)
    laneColor = np.array([0, 255, 0], dtype=np.uint8)
    vpColorMapMat = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    for i in range(0, len(markClass)):
        if i != 0:
            colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

    for imageFile in tqdm(listOfFiles):
        data = scipy.io.loadmat(imageFile)
        rgb_seg_vp = data['rgb_seg_vp']
        rgb = rgb_seg_vp[:, :, 0: 3]
        seg = rgb_seg_vp[:, :, 3]
        vp = rgb_seg_vp[:, :, 4]
        img_bgr = rgb[:, :, :: -1]
        segImage = colorMapMat[seg]
        x = np.nonzero(vp)[1]
        y = np.nonzero(vp)[0]
        if x.size > 0 and y.size > 0:
            vpPoint = (x[0], y[0])
            vpLevel = vp[(y, x)]
        else:
            vpPoint = (-1, -1)

        if vpPoint[0] > 1 and vpPoint[1] > 1:
            c = vpColorMapMat[vpLevel-1][0].tolist()
            cv2.circle(segImage, vpPoint, 15, (c[0], c[1],
                                               c[2]), thickness=-1, lineType=8)

        res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0)
        cv2.imshow('VPGNet Dataset Quick Inspector', res)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyWindow('VPGNet Dataset Quick Inspector')


def parse_args():
    parser = argparse.ArgumentParser(
        description='VPGNet Dataset Quick Inspector')
    parser.add_argument('--rootDir', type=str, default=r'D:\VPGNet-DB-5ch',
                        help='root directory (default: D:\\VPGNet-DB-5ch)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    walkThroughDataset(args.rootDir)
