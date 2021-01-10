import os
import sys
from numpy.testing._private.utils import assert_equal
import pandas as pd
import argparse
from pandas.core.arrays.categorical import Categorical
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def save_tfrecords(df, y, out_dir):
    if len(df)!=len(y):
        raise ValueError(f"The features ({len(df)}) and labels ({len(y)}) are different lengths")
    out_file = os.path.join(out_dir, 'features.tfrecords')
    with tf.io.TFRecordWriter(out_file) as writer:
        for i in range(len(df)):
            feature = df.iloc[i].values
            label = y.iloc[i].values
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
                    }
                )
            )
            writer.write(example.SerializeToString())


def encode_categories(df, categories):
    # remove labels from the data
    labels = df.pop('label')
    # get categorical type first so all notes are represented and in order
    labels = pd.Categorical(labels, categories, ordered=True)
    # one hot encode the labels
    labels = pd.get_dummies(labels)   
    # assert(np.array_equal(le.inverse_transform(y), labels.values) == True)
    return df, labels


if __name__ == "__main__":

    params = yaml.safe_load(open('params.yaml'))['store_features']

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython store_features.py raw_file freq_file out_folder\n")
        sys.exit(1)

    input_file = os.path.abspath(sys.argv[1])
    freq_file = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3])
    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    df = pd.read_csv(input_file)
    df_freq = pd.read_csv(freq_file)
    df, y = encode_categories(df, categories=df_freq['name'].values)

    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

    save_tfrecords(df_train, y_train, train_dir)
    save_tfrecords(df_test, y_test, test_dir)
