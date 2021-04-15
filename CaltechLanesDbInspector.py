import scipy.io as sio
import numpy as np
import cv2
import os
import argparse


def walkThroughDataset(dataSetDir):
    scene_list = ['cordova1', 'cordova2', 'washington1', 'washington2']
    labels_file_name = 'labels.ccvl'

    subtype_colors = {'bw': (255,0,0), 'dy': (0,255,0), 'sw': (0,0,255)}
    for scene_name in scene_list:
        scene_data_path = os.path.join(dataSetDir,scene_name)
        labels_path = os.path.join(scene_data_path,labels_file_name)

        label_data=sio.loadmat(labels_path, simplify_cells=True)
        frames_data = label_data['ld']['frames']
        for frame_data in frames_data:
            frame_name = frame_data['frame']
            img_path = os.path.join(scene_data_path, frame_name)
            img = cv2.imread(img_path)
            label_data = frame_data['labels'] if isinstance(frame_data['labels'], list) else [frame_data['labels']]

            for label in label_data:
                if len(label) !=0:
                    subtype = label['subtype']
                    type = label['type']
                    if type == 'spline':
                        points = label['points']
                        points_new = get_lane_points(spline=points, interval=0.02)

                        # points_new = [(x,y) for x,y in zip(xnew, ynew)]
                        for point in points_new:
                            coord = (int(point[0]), int(point[1]))
                            cv2.circle(img, coord, 4, subtype_colors[subtype], thickness=-1, lineType=8)

            cv2.imshow('Caltech Lanes Dataset Quick Inspector', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            


def get_lane_points(spline, interval=0.05):
    """
    Function to evaluate a Bezier spline with the specified degree.
    Algorithm taken from the matlab code: https://github.com/mohamedadaly/caltechcv-image-labeler/blob/master/ccvEvalBezSpline.m

    Parameters
    ----------
    spline    - input 3 or 4 points in a matrix 3x2 or 4x2 [xs, ys]
    interval         - [0.05] the interval to use for evaluation

    Returns
    -------
    output points nx2 [xs; ys]
    
    """

    degree = spline.shape[0] - 1
    n = int(np.floor(1/interval) + 1)
    if degree == 2:
        M =	np.array([[1, -2, 1], [-2,2,0], [1,0,0]])
        abcd= np.matmul(M, spline)
        a = abcd[0]
        b = abcd[1]
        c = abcd[2]

        P = c
        dP = b*interval + a*interval**2
        ddP = 2*a*interval**2

        outPoints = np.zeros((n, spline.shape[1]))
        outPoints[0,:] = P
        for i in range(1,n):
            # calculate new point
            P = P + dP
            # update steps
            dP = dP + ddP
            # store in output array
            outPoints[i,:] = P
    elif degree == 3:
        M =	np.array([[-1, 3, -3,1], [3,-6,3,0], [-3,3,0,0], [1,0,0,0]])
        abcd= np.matmul(M, spline)
        a = abcd[0]
        b = abcd[1]
        c = abcd[2]
        d = abcd[3]

        P = d
        dP = c*interval + b * interval**2 + a*interval**3
        ddP = 2*b*interval**2 + 6*a*interval**3
        dddP = 6*a*interval**3

        outPoints = np.zeros((n, spline.shape[1]))
        outPoints[0,:] = P
        for i in range(1,n):
			# calculate new point
            P = P + dP
			# update steps
            dP = dP + ddP
            ddP = ddP + dddP
			# store in output array
            outPoints[i,:] = P
    else:
        raise NotImplementedError(f"Supports Bezier spline with degrees 2 or 3, but got {degree}")
		
    return outPoints


def parse_args():
    parser = argparse.ArgumentParser(
        description='Caltech Lanes Dataset Quick Inspector')
    parser.add_argument('--rootDir', type=str, default='/media/caltech-lanes',
                        help='root directory (default: /media/caltech-lanes)')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    walkThroughDataset(args.rootDir)
