# import pickle
#
# pkl_path = ['../ssd300_sixray/test/不带电芯充电宝_pr.pkl', '../ssd300_sixray/test/带电芯充电宝_pr.pkl']
#
# for path in pkl_path:
#     with open(path, 'rb') as f:
#         print(pickle.load(f))

import cv2
import numpy as np
import os

path = 'G:/MachineLearning/cover/test_data/Image'
def compute():
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:,:,0]))
        per_image_Gmean.append(np.mean(img[:,:,1]))
        per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return B_mean, G_mean, R_mean



if __name__ == '__main__':
    img = cv2.imread('G:/MachineLearning/cover/test_data/Image/coreless_battery00000223.jpg')
    #print(compute())
    mean = (1.0, 50.0, 50.0)

    x = cv2.resize(img, (300, 300)).astype(np.float32)
    # x -= mean
    x += (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    #x = cv2.resize(img, (300, 300))
    x = x[:, :, ::-1].copy()
    cv2.imshow('test', x)
    cv2.waitKey()

