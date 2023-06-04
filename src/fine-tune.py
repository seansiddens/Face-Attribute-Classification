import tensorflow as tf
from model import load_pretrained, define_metrics, save_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def parse_data(example_proto):
    """
    Parse tfrecord data

    Parameters
    ----------
    example_proto : A Dataset comprising records from one or more TFRecord files
    size  : Image size of type tuple, should be (height, width)
    model : String representing the base model used (vgg16, inception_v3, or resnet50)

    Return
    ------
        x_train : Preprocessed numpy.array or a tf.Tensor with type float32
        y_train : A dense tensor with shape of (length(label), 1)
    """

    image_feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/label": tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, image_feature_description)
    x_train = tf.image.decode_jpeg(x["image/encoded"])
    x_train = tf.image.resize(x_train, (224, 224))
    x_train = tf.keras.applications.vgg16.preprocess_input(x_train)

    y_train = tf.sparse.to_dense(x["image/label"])
    return x_train, y_train


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


# Add a final binary-classification layer which classifies a single attribute as either
# present or not present.
def method_2():
    # Load pre-trained base model.
    pretrained_model = load_pretrained("src/models/vgg16.h5")

    # Freeze base model's layers.
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Add new binary classifier output layer
    x = pretrained_model.output
    output_layer = Dense(2, activation="sigmoid", name="new-output")(x)

    # Create new model and print summary.
    new_model = Model(inputs=pretrained_model.input, outputs=output_layer)

    # Define training metrics
    metrics = define_metrics()
    new_model.summary()

    # Define the optimizer
    opt = Adam(learning_rate=1e-4)

    # Compile the new model
    new_model.compile(optimizer=opt, metrics=metrics, loss="categorical_crossentropy")

    # Prepare training data.
    batch_size = 8
    train_data = load_dataset("src/tfrecords/bald-train.tfrecord", 8)
    validation_data = load_dataset("src/tfrecords/bald-val.tfrecord", 8)

    # Train new model
    new_model.fit(train_data, epochs=10, validation_data=validation_data, verbose=1)

    # save_model(new_model, 'models', 'fine-tuned')


def main():
    method_2()


if __name__ == "__main__":
    main()
