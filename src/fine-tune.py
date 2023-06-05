import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cv2
import os
import numpy as np
from model import load_pretrained, define_metrics, save_model
from utils import parse_data

ATTRIBUTES = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()

def load_dataset(path, batch_size):
    """
    Load a tfrecord file

    Parameters
    ----------
    path  : Path of tfrecord file
    batch_size : Number of batch size to load the data
    size  : Image size of type tuple, should be (height, width)
    model : String representing the base model used (vgg16, inception_v3, or resnet50)

    Return
    ------
        An Iterator over the elements of the dataset
    """

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(lambda x: parse_data(x))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def create_training_data():
    path = "../new-dataset/bald"
    training_data = []

    # Iterate over every sub-folder in root dataset folder.
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            for img in os.listdir(subfolder_path):
                try:
                    # Read in every image in the folder.
                    img_array = cv2.imread(os.path.join(subfolder_path, img), cv2.IMREAD_COLOR)
                except cv2.error as e:
                    print(f"Error: {e}")

                if img_array is None:
                    print("Error reading image")
                    continue

                # Resize and preprocess image.
                img_array = cv2.resize(img_array, (224, 224))
                img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
                label = [1, 0]
                training_data.append((img_array, label))
        
    # Add negative example training data.
    num_positive_examples = len(training_data)
    print(f"num positive examples: {num_positive_examples}")
    num_negative_examples = 0
    celeba_data = tf.data.TFRecordDataset("src/test.tfrecord")
    celeba_data = celeba_data.map(lambda x: parse_data(x, (224, 224), 'vgg16'))
    print("Adding negative examples to training set.")
    for img, labels in celeba_data:
        # Add new training examples until we have the same number of positive and negative.
        if num_negative_examples >= num_positive_examples:
            break
        
        for i, attr in enumerate(ATTRIBUTES):
            if attr == "Bald":
                if labels[i] != 1:
                    # Example is not bald, add to training set.
                    training_data.append((img, [0, 1]))
                    num_negative_examples += 1

    return training_data


# TODO: This function signature might not match what tensorflow expects.
# Might have to have missing labels have a special value (nan?) such that the loss function
# knows which neurons to take into account.
def custom_loss(y_true, y_pred, selected_neurons):
    # Get the predictions from the selected neurons.
    selected_pred = tf.gather(y_pred, selected_neurons, axis=1)

    # Get the ground truth labels from the selected neurons
    selected_true = tf.gather(y_true, selected_neurons, axis=1)

    # Calculate the loss
    loss = tf.losses.binary_crossentropy(selected_true, selected_pred)

    return loss


# Source: https://stackoverflow.com/a/57915320
def other_custom_loss(y_true, y_pred):
    # We're adding a small epsilon value to prevent computing logarithm of 0 (consider y_hat == 0.0 or y_hat == 1.0).
    epsilon = tf.constant(1.0e-30, dtype=np.float32)

    # Check that there are no NaN values in predictions (neural network shouldn't output NaNs).
    y_pred = tf.debugging.assert_all_finite(y_pred, "y_pred contains NaN")

    # Temporarily replace missing values with zeroes, storing the missing values mask for later.
    y_true_not_nan_mask = tf.logical_not(tf.math.is_nan(y_true))
    y_true_nan_replaced = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true
    )

    # Cross entropy, but split into multiple lines for readability:
    # y * log(y_hat)
    positive_predictions_cross_entropy = y_true_nan_replaced * tf.math.log(
        y_pred + epsilon
    )
    # (1 - y) * log(1 - y_hat)
    negative_predictions_cross_entropy = (1.0 - y_true_nan_replaced) * tf.math.log(
        1.0 - y_pred + epsilon
    )
    # c(y, y_hat) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    cross_entropy_loss = -(
        positive_predictions_cross_entropy + negative_predictions_cross_entropy
    )

    # Use the missing values mask for replacing loss values in places in which the label was missing with zeroes.
    # (y_true_not_nan_mask is a boolean which when casted to float will take values of 0.0 or 1.0)
    cross_entropy_loss_discarded_nan_labels = cross_entropy_loss * tf.cast(
        y_true_not_nan_mask, tf.float32
    )

    mean_loss_per_row = tf.reduce_mean(cross_entropy_loss_discarded_nan_labels, axis=1)
    mean_loss = tf.reduce_mean(mean_loss_per_row)

    return mean_loss

def evaluate_model(model, test_data):
    # print("Testing trained model...")
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    num_positive_examples = 0
    num_negative_examples = 0
    threshold = 0.5
    for img, label in test_data:
        label = list(label)
        if label == [1, 0]:
            # Positive test example
            num_positive_examples += 1
        elif label == [0, 1]:
            num_negative_examples += 1

        # Make a prediction on example.
        pred = model.predict(np.array([img]), verbose=0)

        if pred[0, 0] >= threshold and pred[0, 1] < threshold:
            # Predicted positive class
            predicted_class = 1
        elif pred[0, 0] < threshold and pred[0, 1] >= threshold:
            # Predicted negative class
            predicted_class = 0
        else:
            # Invalid (contradictory) prediction!
            predicted_class = -1

        # Record prediction result.
        if predicted_class == 1 and label == [1, 0]:
            tp += 1
        elif predicted_class == 0 and label == [0, 1]:
            tn += 1
        elif predicted_class == 1 and label == [0, 1]:
            fp += 1
        elif predicted_class == 0 and label == [1, 0]:
            fn += 1
            
        # print(f"Test label: {label}")
        # print(f"Model prediction: {pred}")
        # print(f"Predicted class: {predicted_class}")
        # print()
    
    print(f"Total testing examples: {len(test_data)}")
    print(f"Number of positive examples: {num_positive_examples}")
    print(f"Number of negative examples: {num_negative_examples}")
    
    # Calculate and print metrics.
    print()
    accuracy = (tp + tn) / (tp + tn + fp + fn) # measures overall correctness of model's predictions
    print(f"Accuracy: {accuracy}")
    precision = tp / (tp + fp) # proportion of correctly predicted positives out of all predicted positves
    print(f"Precision: {precision}")
    recall = tp / (tp + fn) # (true pos rate) proportion of correctly predicted positives out of all actual positives
    print(f"Recall (true pos rate): {recall}")
    specificity = tn / (tn + fp) # (true neg rate) proportion of corretly predicted negatives out of all actual negatives
    print(f"Specificity: {specificity}")
    f1 = 2 * ((precision * recall) / (precision + recall)) 
    print(f"F1 Score: {f1}")
    print()


# Add a final binary-classification layer which classifies a single attribute as either
# present or not present.
def method_2():
    # Load pre-trained base model.
    pretrained_model = load_pretrained("src/models/vgg16.h5")

    # # Freeze base model's layers.
    # for layer in pretrained_model.layers:
    #     layer.trainable = False
    for layer in pretrained_model.layers[:-4]:
        # Freeze every layer of base model.
        layer.trainable = False

    # Add new binary classifier output layer
    x = pretrained_model.output
    output_layer = Dense(2, activation="sigmoid", name="new-output")(x)

    # Create new model and print summary.
    new_model = Model(inputs=pretrained_model.input, outputs=output_layer)
    new_model.summary()

    # Define training metrics
    metrics = define_metrics()

    # Define the optimizer
    opt = Adam(learning_rate=1e-4)

    # Compile the new model
    new_model.compile(optimizer=opt, metrics=metrics, loss="categorical_crossentropy")
    # Save untrained model.
    # save_model(new_model, "src/models", "untrained-frozen-base-model")

    # Get image, label pairs
    data = create_training_data()
    print(f"Total data: {len(data)}")
    # Shuffle data
    np.random.shuffle(data)

    # Split data into a training and validation set.
    split_index = int(len(data) * 0.1)
    validation_data = data[:split_index]
    test_data = data[split_index:split_index*2]
    training_data = data[split_index*2:]

    print(f"# validation examples: {len(validation_data)}")
    print(f"# test examples: {len(test_data)}")
    print(f"# training examples: {len(training_data)}")

    # Split into images and labels
    train_images = np.array([data[0] for data in training_data])
    train_labels = np.array([data[1] for data in training_data])
    val_images = np.array([data[0] for data in validation_data])
    val_labels = np.array([data[1] for data in validation_data])
    test_images = np.array([data[0] for data in test_data])
    test_labels = np.array([data[1] for data in test_data])

    # Evaluate un-trained model
    print("Evaluating untrained model...")
    evaluate_model(new_model, test_data)

    # Batch data into training sets
    batch_size = 8
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)

    # Fine tune the model.
    new_model.fit(train_dataset, epochs=10, validation_data=val_dataset, verbose=1)

    # Save fine-tuned model
    # save_model(new_model, "src/models", "fine-tuned-frozen-base-model-equal-split")

    # Evaluate the model.
    print("Evaluating trained model")
    evaluate_model(new_model, test_data)


def main():
    method_2()


if __name__ == "__main__":
    main()
