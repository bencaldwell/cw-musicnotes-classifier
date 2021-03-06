import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np
import tensorflow as tf
import json
import time
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


def input_fn(input_dir, feature_size, num_classes, batch_size):
    feature_description = {
        'feature': tf.io.FixedLenFeature([feature_size], tf.float32),
        'label': tf.io.FixedLenFeature([num_classes], tf.float32),
    }

    def parse(record):
        parsed = tf.io.parse_single_example(record, feature_description)
        return parsed['feature'], parsed['label']

    filenames = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    ds = tf.data.TFRecordDataset(filenames=filenames)
    ds = ds.map(parse)
    ds = ds.shuffle(100, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds


def model(feature_size, num_classes, kernel_size, pool_size, filters):
    model = Sequential()
    model.add(layers.Reshape((feature_size, 1), input_shape=(feature_size,)))
    model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.Flatten())
    model.add(layers.ReLU())
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'categorical_accuracy'
        ]

    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--model_save', type=str)
    parser.add_argument('--model_load', type=str)
    parser.add_argument('--metrics_dir', type=str)
    return parser.parse_known_args()


if __name__ == "__main__":

    args, unknown = parse_args()
    params = yaml.safe_load(open('params.yaml'))['train']

    feature_size = params['feature_size']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    num_classes = int(params['num_classes'])
    kernel_size = int(params['kernel_size'])
    pool_size = int(params['pool_size'])
    filters = int(params['filters'])

    train_dir = os.path.abspath(args.train_dir)
    test_dir = os.path.abspath(args.test_dir)
    metrics_dir = os.path.abspath(args.metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    model_save = os.path.abspath(args.model_save)
    os.makedirs(model_save, exist_ok=True)
    if (args.model_load):
        model_load = os.path.abspath(args.model_load)
    else:
        model_load = None

    ds_list = list(input_fn(train_dir, feature_size, num_classes, batch_size))
    feature_list = []
    label_list = []
    for feature, label in ds_list:
        label_list.append(label.numpy())
        feature_list.append(feature.numpy())
    features = np.vstack(feature_list)
    labels = np.vstack(label_list)
    print(f"train features shape: {features.shape}")
    print(f'train labels: {np.unique(labels, return_counts=True)}')

    try:
        model = tf.keras.models.load_model(model_load)
        print(f'Loading model from {model_load}')
    except:
        print(f'Creating new model')
        model = model(
            feature_size=feature_size,
            num_classes=num_classes,
            kernel_size=kernel_size,
            pool_size=pool_size,
            filters=filters
        )
    print(model.summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        restore_best_weights=True
    )

    start_real = time.time()
    start_process = time.process_time()

    history = model.fit(
        input_fn(train_dir, feature_size, num_classes, batch_size),
        epochs=epochs,
        validation_data=input_fn(
            test_dir, feature_size, num_classes, batch_size),
        callbacks=[early_stopping],
        verbose=2
    )

    end_real = time.time()
    end_process = time.process_time()

    print(f'metrics available: {history.history.keys()}')

    metrics_file = os.path.join(metrics_dir, 'summary.json')
    with open(metrics_file, 'w') as fd:
        json.dump({
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "categorical_accuracy": float(history.history["categorical_accuracy"][-1]),
            "val_categorical_accuracy": float(history.history["val_categorical_accuracy"][-1]),
            "time_real": end_real - start_real,
            "time_process": end_process - start_process
        }, fd)

    model.save(model_save)