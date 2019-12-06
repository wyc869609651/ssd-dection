import os
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


SIXray_CLASSES = (
    'core_battery', 'coreless_battery'
)
# note: if you used our download scripts, this should be right
SIXray_ROOT = "D:/wuyanchen/buaa"


class SIXrayAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            # zip(SIXray_CLASSES, range(len(SIXray_CLASSES))))
            zip(SIXray_CLASSES, range(len(SIXray_CLASSES))))
        self.keep_difficult = keep_difficult
        # 添加的记录所有小类总数
        self.type_dict = {}
        # 记录大类数量
        self.type_sum_dict = {}

    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            it has been changed to the path of annotation-2019-07-10
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # 遍历Annotation
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]
            # 只读两类
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2]) / width
            # 只读取V视角的
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:  # 是这么个意思吧？
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, 3]]
        return res


class SIXrayDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=['core_500', 'coreless_5000'],
                 transform=None, target_transform=SIXrayAnnotationTransform(),
                 dataset_name='SIXray_train'):
        # self.root = root
        self.root = SIXray_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join(self.root, '%s', 'Annotation', '%s.txt')
        self._imgpath = osp.join(self.root, '%s', 'Image', '%s.jpg')
        self.ids = list()
        for dirname in image_sets:
            nameListPath = osp.join(self.root, dirname, 'nameList.txt')
            for line in open(nameListPath, 'r'):
                self.ids.append((dirname, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, og_im = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self._annopath % img_id  # 注释目录
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            print('\nwrong\n')
            print(self._imgpath % img_id)
            return None

        height, width, channels = img.shape
        og_img = img

        # print (img_id)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

