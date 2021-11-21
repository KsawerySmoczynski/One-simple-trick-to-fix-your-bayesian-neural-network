START_TIME=$(date +'%r')
python scripts/train.py --config configs/MNIST/LogisticRegression.yaml 2>&1 | tee logs/01_LogisticRegression.txt
python scripts/train.py --config configs/MNIST/FNN.yaml 2>&1 | tee logs/02_FNN.yaml
python scripts/train.py --config configs/MNIST/ConvFNN.yaml 2>&1 | tee logs/03_ConvFNN.txt
python scripts/train.py --config configs/MNIST/FCN.yaml 2>&1 | tee logs/04_FCN.txt
python scripts/train.py --config configs/MNIST/SeparableFCN.yaml 2>&1 | tee logs/05_SeparableFCN.txt
python scripts/train.py --config configs/MNIST/LeNet.yaml 2>&1 | tee logs/06_LeNet.txt
echo $START_TIME
END_TIME=$(date +'%r')
echo $END_TIME
