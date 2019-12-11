import cv2
import os
import numpy as np

ROOT_PATH = 'G:\\MachineLearning\\unbalance\\core_500'
image_path = os.path.join(ROOT_PATH, 'Image')  # 原图像保存位置
annotation_path = os.path.join(ROOT_PATH, 'Annotation')  # 原目标框保存位置
image_save_path = os.path.join(ROOT_PATH, 'Image_new')  # 原目标框保存位置
annotation_save_path = os.path.join(ROOT_PATH, 'Annotation_new')  # 原目标框保存位置

def img_flip(img_path, annot_path, new_name):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    new_img = cv2.flip(img, 1) #水平翻转图片
    img_name = 'core_battery' + new_name + '.jpg'
    new_img_path = os.path.join(image_save_path, img_name)
    cv2.imwrite(new_img_path, new_img)

    with open(annot_path, encoding='utf-8') as fp:
        lines = fp.readlines()
    annot_str = ''
    for line in lines:
        temp = line.split()
        name = temp[1]
        # 只读两类
        if name != '带电芯充电宝' and name != '不带电芯充电宝':
            continue
        xmin = int(temp[2])
        if xmin > img_w:
            continue
        if xmin < 0:
            xmin = 0
        ymin = int(temp[3])
        if ymin < 0:
            ymin = 0
        xmax = int(temp[4])
        if xmax > img_w:  # 是这么个意思吧？
            xmax = img_w
        ymax = int(temp[5])
        if ymax > img_h:
            ymax = img_h
        annot_w = xmax - xmin
        new_xmin = img_w - xmin - annot_w
        new_ymin = ymin
        new_xmax = img_w - xmax + annot_w
        new_ymax = ymax
        annot_str += temp[0] + ' ' + temp[1] + ' ' + str(new_xmin) + ' ' + str(new_ymin) + ' ' + str(new_xmax) + ' ' + str(new_ymax) + '\n'
    annot_name = 'core_battery' + new_name + '.txt'
    new_annot_path = os.path.join(annotation_save_path, annot_name)
    with open(new_annot_path, 'w', encoding='utf-8') as fp:
        fp.write(annot_str)
    return new_img_path, new_annot_path

def img_rot(img_path, annot_path, new_name):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    # 旋转图像
    if img_w > img_h:
        padding = (img_w - img_h) // 2
        center = (img_w // 2, img_w // 2)
        img_padded = np.zeros(shape=(img_w, img_w, 3), dtype=np.uint8)
        img_padded[padding:padding + img_h, :, :] = img

        M = cv2.getRotationMatrix2D(center, -90, 1)
        rotated_padded = cv2.warpAffine(img_padded, M, (img_w, img_w))
        new_img = rotated_padded[:, padding:padding + img_h, :]
    else:
        padding = (img_h - img_w) // 2
        center = (img_h // 2, img_h // 2)
        img_padded = np.zeros(shape=(img_h, img_h, 3), dtype=np.uint8)
        img_padded[:, padding:padding + img_w, :] = img
        M = cv2.getRotationMatrix2D(center, -90, 1)
        rotated_padded = cv2.warpAffine(img_padded, M, (img_h, img_h))
        new_img = rotated_padded[padding:padding + img_w, :, :]

    img_name = 'core_battery' + new_name + '.jpg'
    new_img_path = os.path.join(image_save_path, img_name)
    cv2.imwrite(new_img_path, new_img)
    with open(annot_path, encoding='utf-8') as fp:
        lines = fp.readlines()
    annot_str = ''
    for line in lines:
        temp = line.split()
        name = temp[1]
        # 只读两类
        if name != '带电芯充电宝' and name != '不带电芯充电宝':
            continue
        xmin = int(temp[2])
        if xmin > img_w:
            continue
        if xmin < 0:
            xmin = 0
        ymin = int(temp[3])
        if ymin < 0:
            ymin = 0
        xmax = int(temp[4])
        if xmax > img_w:  # 是这么个意思吧？
            xmax = img_w
        ymax = int(temp[5])
        if ymax > img_h:
            ymax = img_h
        annot_h = ymax - ymin
        new_xmin = img_h - ymin - annot_h
        new_ymin = xmin
        new_xmax = img_h - ymax + annot_h
        new_ymax = xmax
        annot_str += temp[0] + ' ' + temp[1] + ' ' + str(new_xmin) + ' ' + str(new_ymin) + ' ' + str(new_xmax) + ' ' + str(new_ymax) + '\n'
    annot_name = 'core_battery' + new_name + '.txt'
    new_annot_path = os.path.join(annotation_save_path, annot_name)
    with open(new_annot_path, 'w', encoding='utf-8') as fp:
        fp.write(annot_str)
    return new_img_path, new_annot_path

def show_image(img_path, annot_path):
    with open(annot_path, encoding='utf-8') as fp:
        lines = fp.readlines()
    for line in lines:
        temp = line.split()
        name = temp[1]
        # 只读两类
        if name != '带电芯充电宝' and name != '不带电芯充电宝':
            continue
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
    img = cv2.imread(img_path)
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    img = cv2.resize(img, (500, 500))
    cv2.imshow('img', img)
    cv2.waitKey()

def img_strength():
    if not os.path.isdir(image_save_path):
        os.mkdir(image_save_path)
    if not os.path.isdir(annotation_save_path):
        os.mkdir(annotation_save_path)
    num_i = 10000000
    step = 0
    for txt_name in os.listdir(annotation_path):
        step += 1
        print('step:', str(step))
        annot_path = os.path.join(annotation_path, txt_name)
        img_name = txt_name.replace('.txt', '.jpg')
        img_path = os.path.join(image_path, img_name)
        img_path, annot_path = img_flip(img_path, annot_path, str(num_i))
        num_i += 1
        for i in range(3):
            img_path, annot_path = img_rot(img_path, annot_path, str(num_i))
            num_i += 1
            img_path, annot_path = img_flip(img_path, annot_path, str(num_i))
            num_i += 1

if __name__ == '__main__':
    img_strength()
    # index = 0
    # for txt_name in os.listdir(annotation_save_path)[20:30]:
    #     annot_path = os.path.join(annotation_save_path, txt_name)
    #     img_name = txt_name.replace('.txt', '.jpg')
    #     img_path = os.path.join(image_save_path, img_name)
    #     show_image(img_path, annot_path)




