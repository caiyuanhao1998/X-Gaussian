source activate x_gaussian

python3 train.py -s data/chest.pickle --scene chest

python3 train.py -s data/foot.pickle --scene foot

python3 train.py -s data/abdomen.pickle --scene abdomen

python3 train.py -s data/head.pickle --scene head

python3 train.py -s data/pancreas.pickle --scene pancreas

