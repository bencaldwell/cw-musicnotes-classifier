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
            

def train_input_fn():
    return _input_fn('train')

def test_input_fn():
    return _input_fn('test')

def _input_fn(channel):
    feature_description = {
        'feature': tf.io.FixedLenFeature([feature_size], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }

    def parse(record):
        parsed = tf.io.parse_single_example(record, feature_description)
        return parsed['feature'], parsed['label']


    dir = os.path.join(input_dir, channel)
    filenames = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    ds = tf.data.TFRecordDataset(filenames=filenames)
    ds = ds.map(parse)
    ds = ds.shuffle(100, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds

if __name__ == "__main__":
    
    params = yaml.safe_load(open('params.yaml'))
    global feature_size
    feature_size = params['generate']['sample_freq']
    global batch_size
    batch_size = int(params['train']['batch_size'])

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py in_file model_dir\n")
        sys.exit(1)

    global input_dir
    input_dir = os.path.abspath(sys.argv[1])
    model_dir = os.path.abspath(sys.argv[2])
    model_dir = os.path.join(model_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ds_list = list(train_input_fn())
    feature_list = []
    label_list = []
    for feature, label in ds_list:
        label_list.append(label.numpy())
        feature_list.append(feature.numpy())
    features = np.vstack(feature_list)
    labels = np.vstack(label_list)
    print(f"features shape: {features.shape}")
    print(np.unique(labels, return_counts=True))
    
    # save_tfrecords(df, out_file)
    



    