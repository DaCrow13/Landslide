import sys
from train import train
from test import test
from confusion_matrix import compute_confusion_matrix

if __name__ == '__main__':
    zone = sys.argv[1]
    model_name = sys.argv[2]

    train(zone, model_name)
    test(zone, model_name)
    compute_confusion_matrix(zone, model_name)
