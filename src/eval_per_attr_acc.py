import tensorflow as tf
import numpy as np
import csv
import json
from model import load_pretrained
from utils import parse_data

ATTRIBUTES = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()

pretrained_model = load_pretrained('models/vgg16_2023_05_12_14_32.h5')


def main():
    test_dataset = tf.data.TFRecordDataset('test.tfrecord')
    test_dataset = test_dataset.map(lambda x: parse_data(x, (224, 224), 'vgg16'))

    # Dictionary storing the number of correct predictions made for each attribute.
    correct_pred_count = dict.fromkeys(ATTRIBUTES, 0)

    # Determines threshold for prediction to be considered correct
    threshold = 0.5

    total_test_examples = 0
    for img, labels in test_dataset:
        preds = pretrained_model.predict(np.array([img]))

        for i, attr in enumerate(ATTRIBUTES):
            predicted_class = 1 if float(preds[0, i]) >= threshold else 0
            if predicted_class == labels[i]:
                # Prediction matches true label. 
                correct_pred_count[attr] += 1

        total_test_examples += 1

    # Calculate accuracy for each attribute
    accuracy = {}
    for k, v in correct_pred_count.items():
        accuracy[k] = v / total_test_examples
    
    # Sort by accuracy
    accuracy = dict(sorted(accuracy.items(), key=lambda x: x[1], reverse=True)) 

    # Pretty print results.
    print(f"Total test examples: {total_test_examples}")
    print(json.dumps(accuracy, indent=4))

    # Save to CSV
    out_file = 'accuracy.csv'
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write dict keys as header row
        writer.writerow(accuracy.keys())

        # Write values as next row
        writer.writerow(accuracy.values())

if __name__ == "__main__":
    main()