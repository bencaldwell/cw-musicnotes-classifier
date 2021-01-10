import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def save_tfrecords(df, out_file):
    with tf.io.TFRecordWriter(out_file) as writer:
        for i, row in df.iterrows():
            label = row.pop('label')
            feature = row.values
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                    }
                )
            )
            writer.write(example.SerializeToString())


def train_input_fn(input_dir, feature_size, num_classes, batch_size):
    return _input_fn('train', input_dir, feature_size, num_classes, batch_size)


def test_input_fn(input_dir, feature_size, num_classes, batch_size):
    return _input_fn('test', input_dir, feature_size, num_classes, batch_size)


def _input_fn(channel, input_dir, feature_size, num_classes, batch_size):
    feature_description = {
        'feature': tf.io.FixedLenFeature([feature_size], tf.float32),
        'label': tf.io.FixedLenFeature([num_classes], tf.float32),
    }

    def parse(record):
        parsed = tf.io.parse_single_example(record, feature_description)
        return parsed['feature'], parsed['label']

    dir = os.path.join(input_dir, channel)
    filenames = [os.path.join(dir, f) for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    ds = tf.data.TFRecordDataset(filenames=filenames)
    ds = ds.map(parse)
    ds = ds.shuffle(100, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds


def model(feature_size, num_classes):
    model = Sequential()
    model.add(layers.Reshape((feature_size, 1), input_shape=(feature_size,)))
    model.add(layers.Conv1D(filters=64, kernel_size=3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=100))
    model.add(layers.Flatten())
    model.add(layers.ReLU())
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":

    params = yaml.safe_load(open('params.yaml'))['train']
    feature_size = params['feature_size']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    num_classes = int(params['num_classes'])

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py in_file model_dir\n")
        sys.exit(1)

    input_dir = os.path.abspath(sys.argv[1])
    model_dir = os.path.abspath(sys.argv[2])
    model_dir = os.path.join(model_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ds_list = list(train_input_fn(input_dir, feature_size, num_classes, batch_size))
    feature_list = []
    label_list = []
    for feature, label in ds_list:
        label_list.append(label.numpy())
        feature_list.append(feature.numpy())
    features = np.vstack(feature_list)
    labels = np.vstack(label_list)
    print(f"train features shape: {features.shape}")
    print(f'train labels: {np.unique(labels, return_counts=True)}')

    model = model(feature_size, num_classes)
    print(model.summary())

    history = model.fit(
        train_input_fn(input_dir, feature_size, num_classes, batch_size),
        epochs=epochs,
        validation_data=test_input_fn(input_dir, feature_size, num_classes, batch_size),
        verbose=2
    )
