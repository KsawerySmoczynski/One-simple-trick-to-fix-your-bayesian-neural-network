# Regression
import re

from src.models import classification, regression
from src.models.bnn import BNNClassification, BNNRegression
from src.models.normal import *

CLASSIFICATION_MODELS = [
    classification.ConvClassify,
    classification.DeepMLEClassify,
    classification.MLEClassify,
    classification.LeNet,
    classification.LogisticRegression,
]
REGRESSION_MODELS = [regression.MLERegression]
