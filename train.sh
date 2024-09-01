source activate x_gaussian

python3 train.py -s data/chest.pickle --scene chest  --eval

python3 train.py -s data/foot.pickle --scene foot  --eval

python3 train.py -s data/abdomen.pickle --scene abdomen  --eval

python3 train.py -s data/head.pickle --scene head  --eval

python3 train.py -s data/pancreas.pickle --scene pancreas  --eval

