import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import json
import pickle
from itertools import groupby

BDD100kRootDir = r'H:\BDD100K'

trainJsonFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train.json')
trainPickleFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train.pkl')
trainLaneOnlyPickleFilePath = os.path.join(
    BDD100kRootDir, r'labels\bdd100k_labels_images_train_lane_only.pkl')


def getRandomColour():
    return tuple(np.random.randint(0, 255, dtype=np.uint8, size=3))


test = True

if not test:
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
        json.dump(laneData[np.random.randint(0, 60000)], f, indent=2)


with open('example.json', 'r') as f:
    entry = json.load(f)

imagePath = entry['name']
img_bgr = cv2.imread(imagePath)
labelImage = np.zeros(img_bgr.shape, np.uint8)

for laneItem in entry['labels']:
    curveVerticesDescStr = laneItem['poly2d'][0]['types']
    curveVertices = np.array(laneItem['poly2d'][0]['vertices'], dtype=np.int)
    for vertex1, vertex2 in zip(curveVertices[:-1], curveVertices[1:]):
        color = getRandomColour()
        cv2.line(labelImage, tuple(vertex1),
                 tuple(vertex2), (int(color[0]),int(color[1]),int(color[2])), 3)

    for node,nodeType in zip(curveVertices,curveVerticesDescStr):
        if nodeType=='L':
            cv2.circle(labelImage,tuple(node),5,(255,0,0),-1)
        else:
            cv2.circle(labelImage,tuple(node),5,(0,255,0),-1)

res = cv2.addWeighted(img_bgr, 0.7, labelImage, 1, 0.4)
cv2.imwrite('dsfsf.png', res)
# cv2.imshow('CULane Dataset Quick Inspector', res)
# k = cv2.waitKey(1) & 0xff
# if k == 27:
#     cv2.destroyWindow('CULane Dataset Quick Inspector')

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
