import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np
import tensorflow as tf
import json
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--model_load', type=str)
    parser.add_argument('--predictions_dir', type=str)
    return parser.parse_known_args()


if __name__ == "__main__":

    args, unknown = parse_args()
    params = yaml.safe_load(open('params.yaml'))['validate']

    # feature_size = params['feature_size']
    
    test_dir = os.path.abspath(args.test_dir)
    model_load = os.path.abspath(args.model_load)
    predictions_dir = os.path.abspath(args.predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)

    print('Hello Validate')