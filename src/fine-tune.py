from model import load_pretrained, define_metrics, save_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
    y_pred = tf.debugging.assert_all_finite(y_pred, 'y_pred contains NaN')

    # Temporarily replace missing values with zeroes, storing the missing values mask for later.
    y_true_not_nan_mask = tf.logical_not(tf.math.is_nan(y_true))
    y_true_nan_replaced = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)

    # Cross entropy, but split into multiple lines for readability:
    # y * log(y_hat)
    positive_predictions_cross_entropy = y_true_nan_replaced * tf.math.log(y_pred + epsilon)
    # (1 - y) * log(1 - y_hat)
    negative_predictions_cross_entropy = (1.0 - y_true_nan_replaced) * tf.math.log(1.0 - y_pred + epsilon)
    # c(y, y_hat) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    cross_entropy_loss = -(positive_predictions_cross_entropy + negative_predictions_cross_entropy)

    # Use the missing values mask for replacing loss values in places in which the label was missing with zeroes.
    # (y_true_not_nan_mask is a boolean which when casted to float will take values of 0.0 or 1.0)
    cross_entropy_loss_discarded_nan_labels = cross_entropy_loss * tf.cast(y_true_not_nan_mask, tf.float32)

    mean_loss_per_row = tf.reduce_mean(cross_entropy_loss_discarded_nan_labels, axis=1)
    mean_loss = tf.reduce_mean(mean_loss_per_row)

    return mean_loss 

    

def main():
    pretrained_model = load_pretrained('models/vgg16.h5')

    # Freeze base model's layers.
    for layer in pretrained_model.layers:
        layer.trainable = False
    
    # # Add new top layers
    x = pretrained_model.output
    predictions = Dense(2, activation='sigmoid', name='new-output')(x)
    
    new_model = Model(inputs=pretrained_model.input, outputs=predictions)
    new_model.summary() 

    # Define training metrics
    metrics = define_metrics()

    # Define the optimizer
    opt = Adam(learning_rate = 1e-4)

    # Compile the new model
    new_model.compile(optimizer=opt, metrics=metrics)
    
    save_model(new_model, 'models', 'fine-tuned')


if __name__ == "__main__":
    main()