import tensorflow as tf
import numpy as np
import json
import random
import os
from tqdm import tqdm
from imutils import paths


VAL_SPLIT = 0.2

def convertToJson(imageDir): 
    hash = {}
    imagePaths = list(paths.list_images(imageDir))

    for path in imagePaths: 
        split = path.split("/")
        imageID = split[-1]
        hash[imageID] = [1, 0]

    return hash
    
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example(example):
    encoded = example['encoded']
    label = example['label']
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(encoded),
        'image/label' : _int64_feature(label)
    }))

def main(trainFile, validationFile, datasetPath): 
    #jsonRaw = open(inputFile, "r")
    jsonData = convertToJson(datasetPath) 

    filenames = list(jsonData.keys())
    random.shuffle(filenames)

    train = tf.io.TFRecordWriter(trainFile)
    val = tf.io.TFRecordWriter(validationFile)

    val_total = int(VAL_SPLIT * len(filenames))
    val_num = 0

    for filename in tqdm(filenames, "Loading images"): 
        image_ext = filename.split(".")[1]
        file = open(os.path.join(datasetPath, filename), "rb").read()
        label = list(np.array(jsonData[filename]).astype(int))
        example = {
            'encoded': file,
            'label': label
        }
        # Create tf.Example
        tf_example = create_tf_example(example)
    
        if val_num <= val_total: 
            val.write(tf_example.SerializeToString())
            val_num += 1 
        else: 
            train.write(tf_example.SerializeToString())

    train.close()
    val.close()

    return 0

if __name__ == "__main__": 
    #inputFile = "work-test.json"
    datasetPath = "../bald-dataset/bald-white-man"
    split = datasetPath.split("/")
    validationFile = split[-1] + "-" + "val.tfrecord"
    trainFile = split[-1] + "-" + "train.tfrecord"
    main(trainFile, validationFile, datasetPath)