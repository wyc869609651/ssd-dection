import os


core_500_root = 'G:/MachineLearning/cover/core_3000'
coreless_500_root = 'G:/MachineLearning/cover/coreless_3000'
test_root = 'G:/MachineLearning/cover/test_data'


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