#%%

import random
import pandas as pd
import argparse
import traceback
import numpy as np
import os, sys
from multiprocessing import Process
import pickle
import tensorflow as tf
# %%
with open('/data/hz2529/zion/MVPContext/combined_feature_2021_v2/ENST00000390666.pickle', 'rb') as infl:
    res = pickle.load(infl)
    print(res.shape)
# %%
raw_dataset = tf.data.TFRecordDataset(['/data/hz2529/zion/protein/tf/train_v1.tfrec'])
# %%
k=raw_dataset.take(2)# %%
# %%
def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"label": tf.io.FixedLenFeature([], dtype=tf.string),
        # "feature_dim": tf.io.FixedLenFeature([], dtype=tf.int64),
        "featasd": tf.io.FixedLenFeature([], dtype=tf.string),
        # "enst_name": tf.io.FixedLenFeature([], dtype=tf.string),
        # "protein_len": tf.io.FixedLenFeature([], dtype=tf.float32)
        }
    )
    

#%%%

for raw_record in raw_dataset.take(1):
  print(repr(raw_record))
# %%
j=k.map(decode_fn)
# %%
def list_record_features(tfrecords_path):
    # Dict of extracted feature information
    features = {}
    # Iterate records
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features
# %%
list_record_features('/data/hz2529/zion/protein/tf/train_v1.tfrec')
# %%
k=raw_dataset.take(2).as_numpy_iterator()
j=tf.train.Example.FromString(next(k))
# %%
j['feature']
# %%
tf.parse