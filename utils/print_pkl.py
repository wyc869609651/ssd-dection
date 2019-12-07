import pickle

pkl_path = ['../ssd300_sixray/test/不带电芯充电宝_pr.pkl', '../ssd300_sixray/test/带电芯充电宝_pr.pkl']

for path in pkl_path:
    with open(path, 'rb') as f:
        print(pickle.load(f))