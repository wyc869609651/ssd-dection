import os


core_500_root = 'D:/吴彦辰/buaa/core_500'
coreless_500_root = 'D:/吴彦辰/buaa/coreless_5000'
test_root = 'G:/MachineLearning/test_data1'


def get_train_list():
    pathList = [test_root]
    for _path in pathList:
        nameListPath = os.path.join(_path, 'nameList.txt')
        nameList = open(nameListPath, 'w')
        annoPath = os.path.join(_path, 'annotation')
        for fileName in os.listdir(annoPath):
            nameList.write(fileName.split(".")[0]+'\n')
        nameList.close()

if __name__ == '__main__':
    get_train_list()