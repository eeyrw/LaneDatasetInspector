import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import json
import pickle
from itertools import groupby

TuSimpleRootDir = r'E:\tusimple'

trainJsonFilePath = os.path.join(
    TuSimpleRootDir, r'train_set\label_data_0313.json')


def getRandomColour():
    return tuple(np.random.randint(0, 255, dtype=np.uint8, size=3))




dataList=[]
with open(trainJsonFilePath, 'r') as f:
    for line in f:
        data = json.loads(line)
        dataList.append(data)


colorMapMat = np.zeros((6, 3), dtype=np.uint8)

for i in range(0, 6):
    colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

for dataEntry in tqdm(dataList):

    clipDir=os.path.dirname(dataEntry['raw_file'])
    twentyFrames=[os.path.join(clipDir,'%d.jpg'%x) for x in range(1,21)]
    for frame in twentyFrames:
        img_bgr = cv2.imread(os.path.join(TuSimpleRootDir,'train_set',frame))
        segImage = np.zeros_like(img_bgr)

        for laneIndex,laneXPoints in enumerate(dataEntry['lanes'],0):
            laneLabelColor=colorMapMat[laneIndex]
            
            curveVertices=list(filter(lambda xyPair: xyPair[0]>0 ,zip(laneXPoints,dataEntry['h_samples'])))

            for vertex1, vertex2 in zip(curveVertices[:-1], curveVertices[1:]):
                color = getRandomColour()
                cv2.line(segImage, tuple(vertex1),
                        tuple(vertex2), (int(colorMapMat[5][0]),int(colorMapMat[5][1]),int(colorMapMat[5][2])), 2)

            for node in curveVertices:
                cv2.circle(segImage,tuple(node),5,(int(laneLabelColor[0]),int(laneLabelColor[1]),int(laneLabelColor[2])),-1)


        res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0.4)
        cv2.imshow('TuSimple Dataset Quick Inspector', res)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
cv2.destroyWindow('TuSimple Dataset Quick Inspector')
