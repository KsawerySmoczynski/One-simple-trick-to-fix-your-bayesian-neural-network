mkdir -p logs
START_TIME=$(date +'%r')
python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/LogisticRegression.yaml 2>&1 | tee logs/01_LogisticRegression.txt
python scripts/train.py fit --config configs/base_config.yaml --config configs/MNIST/LeNet.yaml 2>&1 | tee logs/06_LeNet.txt
echo $START_TIME
END_TIME=$(date +'%r')
echo $END_TIME
