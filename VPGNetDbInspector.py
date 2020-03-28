import scipy.io
import numpy as np
import cv2
import os


# Get the list of all files in directory tree at given path
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk('.'):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
fast = cv2.FastFeatureDetector_create(threshold=100)
colorMapMat=np.zeros((18,3),dtype=np.uint8)
laneColor=np.array([0,255,0],dtype=np.uint8)
for i in range(0,18):
    if i>=1 and i<=7:
        colorMapMat[i]=laneColor
    elif i!=0:
        colorMapMat[i]=np.random.randint(0,255,dtype=np.uint8,size=3)

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )    
for imageFile in listOfFiles[1:-1]:
    data = scipy.io.loadmat(imageFile)  # 读取mat文件
    rgb_seg_vp = data['rgb_seg_vp']
    rgb = rgb_seg_vp[:, :, 0:3]
    seg = rgb_seg_vp[:, :, 3]
    vp = rgb_seg_vp[:, :, 4]
    img_bgr = rgb[:, :, ::-1]
    segImage=colorMapMat[seg]
    # print(segImage)
    # Initiate FAST object with default values

    # find and draw the keypoints
    kp = fast.detect(img_bgr,None)
    # img2 = cv2.drawKeypoints(img_bgr, kp, None, color=(255,0,0))
    res = cv2.addWeighted(img_bgr, 1, segImage, 0.6, 0)
    cv2.imshow('BGRimage', res)
    cv2.waitKey(70)
cv2.destroyWindow('BGRimage')
