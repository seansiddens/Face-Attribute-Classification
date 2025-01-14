import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from model import load_pretrained
import csv
import json
import torchvision.transforms as transforms
from imutils import paths
from PIL import Image

ATTRIBUTES = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()


def main(imageFolder, modelPath, predictionFilename): 
    errorFilename = "error-log.txt"
    imagePaths = list(paths.list_images(imageFolder))
    model = load_pretrained(modelPath)

    errorFile = open(errorFilename, "w")

    #create first row of csv 
    predictCSV = open(predictionFilename, "w") 
    headerNames = ["img_id"]
    headerNames = headerNames + ATTRIBUTES
    csvWriter = csv.DictWriter(predictCSV, fieldnames=headerNames)
    csvWriter.writeheader()

    #convert a few images 
    for path in imagePaths : 
        try: 
            # Grab image meta data
            pathCopy = path
            pathCopy = pathCopy.split("/")

            imgNum = ""  
            imgNum = pathCopy[-1] 

            # Load image
            img = np.array(load_img(path))

            # Transform image 
            trans_img = tf.image.resize(img, (224, 224))
            trans_img = tf.keras.applications.vgg16.preprocess_input(trans_img)

            # Make prediction
            preds = model.predict(np.array([trans_img]))

            # Convert output
            predictions = {}
            predictions["img_id"] = imgNum 
            for i, attr in enumerate(ATTRIBUTES):
                predictions[attr] = float(preds[0, i])

            # Pretty print results.

            # Write Results
            csvWriter.writerow(predictions)

        except: 
            errorFile.write(f"{path}\n")


if __name__ == "__main__": 
    imageFolder = "../white-data"
    modelPath = "./models/vgg16.h5"
    predictionFilename = "test.csv"
    main(imageFolder, modelPath, predictionFilename)