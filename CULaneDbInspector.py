import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm


CULaneRootDir = r'H:\CULane'
trainIndexFilePath = os.path.join(CULaneRootDir, r'list\train_gt.txt')

trainFilePairList = []

with open(trainIndexFilePath, 'r') as f:
    for line in f.readlines():
        imagePath, segImagePath, lane0, lane1, lane2, lane4 = line.split()
        trainFilePairList.append(
            (os.path.join(CULaneRootDir, imagePath[1:]), os.path.join(CULaneRootDir, segImagePath[1:]), lane0, lane1, lane2, lane4))

colorMapMat = np.zeros((5, 3), dtype=np.uint8)

for i in range(0, 5):
    if i != 0:
        colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

for imageFile, segFile, _, _, _, _ in tqdm(trainFilePairList):
    img_bgr = cv2.imread(imageFile)
    seg = cv2.imread(segFile,cv2.IMREAD_UNCHANGED)
    # seg = np.max(seg,axis=2)
    segImage = colorMapMat[seg]

    res = cv2.addWeighted(img_bgr, 0.7, segImage, 0.7, 0.4)
    cv2.imshow('CULane Dataset Quick Inspector', res)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cv2.destroyWindow('CULane Dataset Quick Inspector')
