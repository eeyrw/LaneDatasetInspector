import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import json
import pickle

BDD100kRootDir = r'H:\BDD100K'

trainJsonFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train.json')
trainPickleFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train.pkl')
trainLaneOnlyPickleFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train_lane_only.pkl')




if not os.path.isfile(trainLaneOnlyPickleFilePath):
    if not os.path.isfile(trainPickleFilePath):
        with open(trainJsonFilePath, 'r') as f:
            data = json.load(f)
        with open(trainPickleFilePath, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(trainPickleFilePath, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
    for entry in data:
        entry['labels'] = list(filter(
            lambda x: x['category'] == 'lane', entry['labels']))
        laneData.append(entry)

    with open(trainLaneOnlyPickleFilePath, 'wb') as f:
        pickle.dump(laneData, f)
else:
    with open(trainLaneOnlyPickleFilePath, 'rb') as f:
        laneData = pickle.load(f)

# 写入 JSON 数据
with open('example.json', 'w') as f:
    json.dump(laneData[np.random.randint(0,60000)], f,indent=2)
# trainIndexFilePath = os.path.join(CULaneRootDir, r'list\train_gt.txt')

# trainFilePairList = []

# with open(trainIndexFilePath, 'r') as f:
#     for line in f.readlines():
#         imagePath, segImagePath, lane0, lane1, lane2, lane4 = line.split()
#         trainFilePairList.append(
#             (os.path.join(CULaneRootDir, imagePath[1:]), os.path.join(CULaneRootDir, segImagePath[1:]), lane0, lane1, lane2, lane4))

# colorMapMat = np.zeros((5, 3), dtype=np.uint8)

# for i in range(0, 5):
#     if i != 0:
#         colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

# for imageFile, segFile, _, _, _, _ in tqdm(trainFilePairList):
#     img_bgr = cv2.imread(imageFile)
#     seg = cv2.imread(segFile,cv2.IMREAD_UNCHANGED)
#     # seg = np.max(seg,axis=2)
#     segImage = colorMapMat[seg]

#     res = cv2.addWeighted(img_bgr, 0.7, segImage, 0.7, 0.4)
#     cv2.imshow('CULane Dataset Quick Inspector', res)
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break
# cv2.destroyWindow('CULane Dataset Quick Inspector')
