import os
import numpy as np
import spectral
import glob

# =============================================================================
# --- PRE-REQUISITES (ASSUMED TO BE DEFINED ELSEWHERE) ---
# =============================================================================
# This script assumes that the following variables have already been loaded and
# are available in the script's scope. You would typically load your trained
# model and associated objects here.

# --- Placeholder definitions ---
# In your actual code, you would load your real model, scaler, and labels.
# class MockModel:
#     def predict(self, data): return np.random.rand(len(data), 5)
# class MockScaler:
#     def transform(self, data): return data

# model = MockModel()
# scaler = MockScaler()
# y_plant_labels = ['Barley', 'Wheat', 'Canola']
# y_age_labels = ['Early', 'Mid', 'Late']
# y_part_labels = ['Leaf', 'Stem']
# y_health_labels = ['Healthy', 'Stressed']
# y_lifecycle_labels = ['Vegetative', 'Flowering']
# TASK_NAMES = ['plant', 'age', 'part', 'health', 'lifecycle']

# # This is the prediction function from your code, which we assume exists.
# def predict_spectra(batch_spectra, model, scaler, label_maps, task_names):
#     """A placeholder function to simulate the prediction process."""
#     # In your real code, this would contain:
#     # scaled_spectra = scaler.transform(batch_spectra)
#     # predictions = model.predict(scaled_spectra)
#     # ... and then format the output.
#     # For this example, we'll just return random classifications.
#     num_predictions = len(batch_spectra)
#     formatted_predictions = []
#     for _ in range(num_predictions):
#         pred_dict = {
#             'plant': np.random.choice(label_maps['plant']),
#             'age': np.random.choice(label_maps['age']),
#             'part': np.random.choice(label_maps['part']),
#             'health': np.random.choice(label_maps['health']),
#             'lifecycle': np.random.choice(label_maps['lifecycle']),
#         }
#         formatted_predictions.append(pred_dict)
#     return formatted_predictions


def predict_spectra(new_spectra, model, scaler, label_maps, task_names):
    """
    Predicts classifications for multiple tasks for one or more input spectra.

    Args:
        new_spectra (np.ndarray): A NumPy array containing the spectrum/spectra.
                                   Shape should be (num_bands,) for a single spectrum,
                                   or (num_samples, num_bands) for multiple spectra.
        model (tf.keras.Model): The trained Keras model.
        scaler (sklearn.preprocessing.StandardScaler): The StandardScaler *already fitted*
                                                      on the training data.
        label_maps (dict): Dictionary mapping task names (e.g., 'plant') to their
                           corresponding array of string labels (e.g., y_plant_labels).
        task_names (list): List of task names (e.g., ['plant', 'age', ...]).

    Returns:
        list or dict:
            - If a single spectrum was input: A dictionary where keys are task names
              and values are the predicted string labels (e.g., {'plant': 'Rosa_rugosa', 'age': 'M', ...}).
            - If multiple spectra were input: A list of dictionaries, where each
              dictionary represents the predictions for one input spectrum.
        None: If input shape is invalid.

    Raises:
        ValueError: If the number of bands in new_spectra doesn't match the scaler.
    """
    # --- Input Validation and Preparation ---
    if not isinstance(new_spectra, np.ndarray):
        new_spectra = np.array(new_spectra)

    if new_spectra.ndim == 1:
        # Single spectrum provided, reshape to (1, num_bands) for scaler and model
        num_bands = new_spectra.shape[0]
        spectra_batch = new_spectra.reshape(1, -1)
        single_input = True
    elif new_spectra.ndim == 2:
        # Batch of spectra provided
        num_bands = new_spectra.shape[1]
        spectra_batch = new_spectra
        single_input = False
    else:
        print(f"Error: Input spectra must be 1D or 2D, but got {new_spectra.ndim} dimensions.")
        return None

    # Check if number of bands matches the scaler
    if num_bands != scaler.n_features_in_:
        raise ValueError(f"Input spectrum has {num_bands} bands, but the model/scaler "
                         f"was trained with {scaler.n_features_in_} bands.")

    # --- Preprocessing ---
    # 1. Scale using the *fitted* scaler
    spectra_scaled = scaler.transform(spectra_batch)

    # 2. Reshape for Conv1D input: (batch_size, steps=num_bands, channels=1)
    spectra_reshaped = spectra_scaled[..., np.newaxis]

    # --- Prediction ---
    # Get raw probability outputs from the model
    predictions_raw = model.predict(spectra_reshaped)
    # Ensure predictions_raw is a dict (it should be for multi-output)
    if not isinstance(predictions_raw, dict):
         output_layer_names = model.output_names
         predictions_raw = dict(zip(output_layer_names, predictions_raw))


    # --- Output Processing ---
    results = []
    num_samples = spectra_reshaped.shape[0]

    for i in range(num_samples): # Loop through each spectrum in the batch
        sample_predictions = {}
        for task in task_names:
            output_name = f"{task}_output" # e.g., 'plant_output'

            if output_name not in predictions_raw:
                 print(f"Warning: Output key '{output_name}' not found in model predictions for task '{task}'. Skipping.")
                 sample_predictions[task] = "Error: Output not found"
                 continue

            # Get probabilities for the current task and current sample
            task_probs = predictions_raw[output_name][i]

            # Find the index of the highest probability
            predicted_index = np.argmax(task_probs)

            # Convert index back to string label
            try:
                predicted_label = label_maps[task][predicted_index]
            except IndexError:
                predicted_label = f"Error: Index {predicted_index} out of bounds for task '{task}' labels"
            except KeyError:
                predicted_label = f"Error: Task '{task}' not found in label_maps"

            sample_predictions[task] = predicted_label
            # Optional: Add the probability of the predicted class
            sample_predictions[f"{task}_probability"] = float(task_probs[predicted_index])

        results.append(sample_predictions)

    # Return a single dict if only one spectrum was input, otherwise the list
    return results[0] if single_input else results

# =============================================================================
# --- CORE FUNCTION TO PROCESS A SINGLE IMAGE ---
# =============================================================================

def classify_and_save_image(fname_hdr, output_dir, model, scaler, label_maps, task_names):
    """
    Opens, classifies, and saves results for a single ENVI image.

    Args:
        fname_hdr (str): Path to the input ENVI header file (.hdr).
        output_dir (str): Path to the directory where results will be saved.
        model: The trained model object.
        scaler: The fitted scaler object.
        label_maps (dict): Dictionary mapping task names to their labels.
        task_names (list): List of task names.
    """
    print("\n" + "="*80)
    print(f"--- Processing Image: {os.path.basename(fname_hdr)} ---")
    print("="*80)

    try:
        # --- 1. Open the image and read into an array ---
        im = spectral.envi.open(fname_hdr)
        im.Arr = im.load()
        im.List = np.reshape(im.Arr, (im.nrows * im.ncols, im.nbands))
        
        valid_pixel_mask = np.sum(im.List, axis=1) > 0
        valid_spectra = im.List[valid_pixel_mask, :]
        valid_pixel_indices = np.where(valid_pixel_mask)[0]
        n_valid_pixels = len(valid_spectra)

        if n_valid_pixels == 0:
            print("Warning: No valid (non-zero) pixels found in this image. Skipping.")
            return

        print(f"Found {n_valid_pixels} valid pixels to classify.")

        # --- 2. Prepare for Batch Prediction ---
        #PREDICTION_BATCH_SIZE = 262144
        PREDICTION_BATCH_SIZE = 524288
        label_to_int_maps = {task: {label: i for i, label in enumerate(labels)} for task, labels in label_maps.items()}

        classification_maps_flat = {task: np.full(im.nrows * im.ncols, -1, dtype=np.int16) for task in task_names}

        # --- 3. Run Prediction in Batches ---
        print(f"Starting prediction with a batch size of {PREDICTION_BATCH_SIZE}...")
        num_batches = int(np.ceil(n_valid_pixels / PREDICTION_BATCH_SIZE))

        for i in range(num_batches):
            start_idx = i * PREDICTION_BATCH_SIZE
            end_idx = min((i + 1) * PREDICTION_BATCH_SIZE, n_valid_pixels)
            print(f"  Processing batch {i+1}/{num_batches}...")

            batch_spectra = valid_spectra[start_idx:end_idx]
            batch_predictions = predict_spectra(batch_spectra, model, scaler, label_maps, task_names)
            global_indices_for_batch = valid_pixel_indices[start_idx:end_idx]

            for task in task_names:
                label_to_int = label_to_int_maps[task]
                predicted_labels = [p[task] for p in batch_predictions]
                predicted_ints = np.array([label_to_int.get(label, -1) for label in predicted_labels], dtype=np.int16)
                classification_maps_flat[task][global_indices_for_batch] = predicted_ints
        
        print("Prediction complete.")

        # --- 4. Save the Classification Maps ---
        print("Saving classification maps...")
        base_name = os.path.splitext(os.path.basename(fname_hdr))[0]

        for task in task_names:
            flat_map = classification_maps_flat[task]
            reshaped_map = np.reshape(flat_map, (im.nrows, im.ncols))
            output_filename = os.path.join(output_dir, f"{base_name}_{task}_classification.hdr")
            class_names = label_maps[task]
            
            spectral.envi.save_classification(output_filename, reshaped_map, metadata=im.metadata, class_names=class_names)
            print(f"  -> Successfully saved to: {output_filename}")

        # --- 5. Explicitly release memory ---
        # This helps ensure the large image array is cleared before the next loop iteration.
        del im, valid_spectra, classification_maps_flat
        import gc
        gc.collect()

    except Exception as e:
        print(f"\nERROR: Could not process file {fname_hdr}.")
        print(f"Details: {e}\n")


# =============================================================================
# --- MAIN SCRIPT EXECUTION ---
# =============================================================================

def batch_classify(input_source, output_dir, model, scaler, label_maps, task_names):
    """
    Processes ENVI images from a single file, a directory, or a list of files.

    Args:
        input_source (str or list): Can be one of:
                                    - A path to a single ENVI .hdr file.
                                    - A path to a directory containing ENVI .hdr files.
                                    - A list of paths to one or more ENVI .hdr files.
        output_dir (str): Path to the directory where results will be saved.
        model, scaler, label_maps, task_names: Objects required for prediction.
    """
    # --- 1. Validate paths and create output directory ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: '{os.path.abspath(output_dir)}'")

    # --- 2. Determine input type and get list of files ---
    envi_files = []
    if isinstance(input_source, list):
        # Input is a list of file paths
        envi_files = [f for f in input_source if os.path.isfile(f) and f.lower().endswith('.hdr')]
        invalid_files = [f for f in input_source if f not in envi_files]
        if invalid_files:
            print(f"Warning: The following paths were invalid or not .hdr files and will be skipped: {invalid_files}")
    elif isinstance(input_source, str):
        if os.path.isdir(input_source):
            # Input is a directory path
            print(f"Searching for ENVI images in directory: '{input_source}'")
            search_pattern = os.path.join(input_source, '*.hdr')
            envi_files = glob.glob(search_pattern)
        elif os.path.isfile(input_source):
            # Input is a single file path
            if input_source.lower().endswith('.hdr'):
                envi_files = [input_source]
            else:
                print(f"Warning: Input file is not an ENVI header (.hdr) file: {input_source}")
        else:
            print(f"Error: Input path '{input_source}' is not a valid file or directory.")
            return
    else:
        print(f"Error: 'input_source' must be a directory path (string) or a list of file paths. Got: {type(input_source)}")
        return

    if not envi_files:
        print("No valid ENVI header (.hdr) files found to process.")
        return

    print(f"\nFound {len(envi_files)} ENVI images to process.")

    # --- 3. Loop through each file and process it ---
    for hdr_path in envi_files:
        try:
            classify_and_save_image(hdr_path, output_dir, model, scaler, label_maps, task_names)
        except Exception as e:
            print(f"An unexpected error occurred while processing {hdr_path}: {e}")

    print("\n" + "="*80)
    print("--- Batch processing complete for all images. ---")
    print("="*80)


# if __name__ == '__main__':
#     # --- CONFIGURE YOUR DIRECTORIES HERE ---
#     INPUT_DIRECTORY = 'path/to/your/envi_images'
#     OUTPUT_DIRECTORY = 'path/to/your/classification_outputs'

#     # --- LOAD YOUR MODEL AND OTHER OBJECTS HERE ---
#     # model = ...
#     # scaler = ...
#     # label_maps = ...
#     # task_names = ...
    
#     # Run the main function
#     batch_classify(
#         INPUT_DIRECTORY, OUTPUT_DIRECTORY, model, scaler, label_maps, task_names
#     )