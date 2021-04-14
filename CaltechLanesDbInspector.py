import scipy.io as sio
import numpy as np
import cv2
import os
from tqdm import tqdm
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
                        for point in points:
                            coord = (int(point[0]), int(point[1]))
                            cv2.circle(img, coord, 5, subtype_colors[subtype], thickness=-1, lineType=8)

            cv2.imshow('Caltech Lanes Dataset Quick Inspector', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            

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
