mkdir -p logs
START_TIME=$(date +'%r')
# python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/LogisticRegression.yaml 2>&1 | tee logs/01_LogisticRegression.txt
# python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/FCNN.yaml 2>&1 | tee logs/02_FCNN.yaml
# python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/ConvFNN.yaml 2>&1 | tee logs/03_ConvFNN.txt
# python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/FCCN.yaml 2>&1 | tee logs/04_FCCN.txt
# python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/SeparableFCN.yaml 2>&1 | tee logs/05_SeparableFCN.txt
python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/LeNet.yaml 2>&1 | tee logs/06_LeNet.txt
echo $START_TIME
END_TIME=$(date +'%r')
echo $END_TIME
