import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np
import tensorflow as tf

def save_tfrecords(df, out_file):
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
    out_folder = os.path.abspath(sys.argv[2])
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, 'features.tfrecords')
    df = pd.read_csv(input_file)

    df = encode_categories(df)

    save_tfrecords(df, out_file)
    



    