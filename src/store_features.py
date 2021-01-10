import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def save_tfrecords(df, out_dir):
    out_file = os.path.join(out_dir, 'features.tfrecords')
    with tf.io.TFRecordWriter(out_file) as writer:
        for i, row in df.iterrows():
            label = row.pop('label')
            feature = row.values
            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                    }
                )
            )
            writer.write(example.SerializeToString())
            
def encode_categories(df):
    df['label'] = pd.Categorical(df['label'])
    df['label'] = df['label'].cat.codes
    df.head()
    return df

if __name__ == "__main__":
    
    params = yaml.safe_load(open('params.yaml'))['store_features']

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython store_features.py out_folder\n")
        sys.exit(1)

    input_file = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    df = encode_categories(df)

    df_train, df_test = train_test_split(df, test_size=0.2)

    save_tfrecords(df_train, train_dir)
    save_tfrecords(df_test, test_dir)
    



    