import os
import csv
import pandas as pd
import tensorflow as tf

#class PrintMeanPredictionCallback(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        tf.print(epoch," ",self.model.MAE_train)


import tensorflow as tf

class PrintMeanPredictionCallback(tf.keras.callbacks.Callback):
    """
    Callback that writes all custom metrics stored in the 'MAE_train' dictionary
    to a file by converting their values to NumPy arrays.
    """
    def __init__(self, filename='training_metrics.txt'):
        super().__init__()
        self.filename = filename

    def on_epoch_end(self, epoch: int, logs=None):
        # Retrieve the dictionary containing custom metrics (symbolic tensors)
        metrics_dict = self.model.MAE_train
        # Create a new dictionary to hold the converted NumPy values
        metrics_numpy = {}

        # Iterate through each key-value pair in the metrics dictionary
        for key, value in metrics_dict.items():
            try:
                metrics_numpy[key] = value.numpy()
            except Exception:
                # Fallback: use tf.keras.backend.get_value() if .numpy() fails
                metrics_numpy[key] = tf.keras.backend.get_value(value)

        # Build a message with epoch number and all custom metrics with their keys
        msg = f"Epoch {epoch} | Training Metrics:\n"
        for key, val in metrics_numpy.items():
            msg += f"  {key}: {val}\n"
        
        # Append the message to the specified file
        with open(self.filename, 'a') as f:
            f.write(msg)


class CustomSaveCallback(tf.keras.callbacks.Callback):
    """
    Callback that saves weights and biases in a CSV format suitable for 
    Turbomole, at specified intervals.
    """

    def __init__(self, base_output_path: str = "weights_epoch", save_frequency: 
                 int = 200):
        """
        Initialize the CustomSaveCallback.

        Parameters
        ----------
        base_output_path : str
            The filename prefix for the output files. Defaults to 
            "weights_epoch".
        save_frequency : int
            How often to save weights (in epochs). Defaults to saving 
            every 200 epochs.
        """
        super().__init__()
        self.base_output_path = base_output_path
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Called at the end of each epoch. Saves the model weights 
        if the epoch number is a multiple of 'save_frequency'.

        Parameters
        ----------
        epoch : int
            Index of the current training epoch.
        logs : dict
            Currently unused in this callback.
        """
        if epoch % self.save_frequency == 0:
            self._save_weights_turbomole(epoch)

    def _save_weights_turbomole(self, epoch: int):
        """
        Save model weights to a file in a CSV format (compatible with Turbomole).

        Parameters
        ----------
        epoch : int
            Index of the current training epoch.
        """
        # Gather all weights and biases from each layer
        weights_and_biases = [layer.get_weights() for layer in self.model.layers]

        # Flatten them into a single list of 1D arrays (columns of weights, 
        # then biases)
        all_weights = []
        for layer_index, layer_params in enumerate(weights_and_biases, start=1):
            if len(layer_params) == 2:
                # Typically [weights, biases]
                weights, biases = layer_params
                # Each column of 'weights' goes as a separate row 
                # in the final file
                for weight_column in weights.T:
                    all_weights.append(weight_column)
                # Add biases as one row
                all_weights.append(biases)

        # Prepare temporary and final filenames
        tmp_filename = f"tmp_epoch_{epoch+1}"
        final_filename = f"{self.base_output_path}_{epoch+1}"

        # Save to a temporary CSV
        pd.DataFrame(all_weights).to_csv(
            tmp_filename, index=False, header=False, float_format="%24.17E"
        )

        # Reformat the temporary CSV into single-column rows
        with open(tmp_filename, mode='r', encoding='utf-8') as source_file, \
             open(final_filename, mode='w', encoding='utf-8', newline='') as output_file:

            csv_reader = csv.reader(source_file)
            csv_writer = csv.writer(output_file)

            for row in csv_reader:
                for value in row:
                    # Write one value per line
                    if value.strip():
                        csv_writer.writerow([value.strip()])

        # Remove the temporary file
        os.remove(tmp_filename)


class SaveTensorflowCallback(tf.keras.callbacks.Callback):
    """
    Callback that saves model weights in TensorFlow's ".h5" 
    format at specified intervals.
    """

    def __init__(self, base_output_path: str = "weights_epoch", 
                 save_frequency: int = 200):
        """
        Initialize the SaveTensorflowCallback.

        Parameters
        ----------
        base_output_path : str
            The filename prefix for the output files. 
            Defaults to "weights_epoch".
        save_frequency : int
            How often to save weights (in epochs). 
            Defaults to saving every 200 epochs.
        """
        super().__init__()
        self.base_output_path = base_output_path
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Called at the end of each epoch. Saves the model weights if 
        the epoch number is a multiple of 'save_frequency'.

        Parameters
        ----------
        epoch : int
            Index of the current training epoch.
        logs : dict
            Currently unused in this callback.
        """
        if epoch % self.save_frequency == 0:
            self._save_weights_tensorflow(epoch)

    def _save_weights_tensorflow(self, epoch: int): 
        """
        Save model weights in TensorFlow's ".h5" format.

        Parameters
        ----------
        epoch : int
            Index of the current training epoch.
        """
        filename = f"{self.base_output_path}_{epoch+1}.weights.h5"
        self.model.save_weights(filename)
