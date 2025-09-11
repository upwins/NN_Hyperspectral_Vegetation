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
class MockModel:
    def predict(self, data): return np.random.rand(len(data), 5)
class MockScaler:
    def transform(self, data): return data

model = MockModel()
scaler = MockScaler()
y_plant_labels = ['Barley', 'Wheat', 'Canola']
y_age_labels = ['Early', 'Mid', 'Late']
y_part_labels = ['Leaf', 'Stem']
y_health_labels = ['Healthy', 'Stressed']
y_lifecycle_labels = ['Vegetative', 'Flowering']
TASK_NAMES = ['plant', 'age', 'part', 'health', 'lifecycle']

# This is the prediction function from your code, which we assume exists.
def predict_spectra(batch_spectra, model, scaler, label_maps, task_names):
    """A placeholder function to simulate the prediction process."""
    # In your real code, this would contain:
    # scaled_spectra = scaler.transform(batch_spectra)
    # predictions = model.predict(scaled_spectra)
    # ... and then format the output.
    # For this example, we'll just return random classifications.
    num_predictions = len(batch_spectra)
    formatted_predictions = []
    for _ in range(num_predictions):
        pred_dict = {
            'plant': np.random.choice(label_maps['plant']),
            'age': np.random.choice(label_maps['age']),
            'part': np.random.choice(label_maps['part']),
            'health': np.random.choice(label_maps['health']),
            'lifecycle': np.random.choice(label_maps['lifecycle']),
        }
        formatted_predictions.append(pred_dict)
    return formatted_predictions

# =============================================================================
# --- CORE FUNCTION TO PROCESS A SINGLE IMAGE ---
# =============================================================================

def classify_and_save_image(fname_hdr, output_dir):
    """
    Opens, classifies, and saves results for a single ENVI image.

    Args:
        fname_hdr (str): Path to the input ENVI header file (.hdr).
        output_dir (str): Path to the directory where results will be saved.
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
        PREDICTION_BATCH_SIZE = 32768
        label_maps = {'plant': y_plant_labels, 'age': y_age_labels, 'part': y_part_labels, 'health': y_health_labels, 'lifecycle': y_lifecycle_labels}
        label_to_int_maps = {task: {label: i for i, label in enumerate(labels)} for task, labels in label_maps.items()}

        classification_maps_flat = {task: np.full(im.nrows * im.ncols, -1, dtype=np.int16) for task in TASK_NAMES}

        # --- 3. Run Prediction in Batches ---
        print(f"Starting prediction with a batch size of {PREDICTION_BATCH_SIZE}...")
        num_batches = int(np.ceil(n_valid_pixels / PREDICTION_BATCH_SIZE))

        for i in range(num_batches):
            start_idx = i * PREDICTION_BATCH_SIZE
            end_idx = min((i + 1) * PREDICTION_BATCH_SIZE, n_valid_pixels)
            print(f"  Processing batch {i+1}/{num_batches}...")

            batch_spectra = valid_spectra[start_idx:end_idx]
            batch_predictions = predict_spectra(batch_spectra, model, scaler, label_maps, TASK_NAMES)
            global_indices_for_batch = valid_pixel_indices[start_idx:end_idx]

            for task in TASK_NAMES:
                label_to_int = label_to_int_maps[task]
                predicted_labels = [p[task] for p in batch_predictions]
                predicted_ints = np.array([label_to_int.get(label, -1) for label in predicted_labels], dtype=np.int16)
                classification_maps_flat[task][global_indices_for_batch] = predicted_ints
        
        print("Prediction complete.")

        # --- 4. Save the Classification Maps ---
        print("Saving classification maps...")
        base_name = os.path.splitext(os.path.basename(fname_hdr))[0]

        for task in TASK_NAMES:
            flat_map = classification_maps_flat[task]
            reshaped_map = np.reshape(flat_map, (im.nrows, im.ncols))
            output_filename = os.path.join(output_dir, f"{base_name}_{task}_classification.hdr")
            class_names = label_maps[task]
            
            spectral.envi.save_classification(output_filename, reshaped_map, metadata=im.metadata, class_names=class_names)
            print(f"  -> Successfully saved to: {output_filename}")

    except Exception as e:
        print(f"\nERROR: Could not process file {fname_hdr}.")
        print(f"Details: {e}\n")


# =============================================================================
# --- MAIN SCRIPT EXECUTION ---
# =============================================================================

def batch_classify_directory(input_dir, output_dir):
    """
    Finds all ENVI images in a directory and runs the classification process on them.
    """
    # --- 1. Validate paths and create output directory ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: '{os.path.abspath(output_dir)}'")

    # --- 2. Find all ENVI header files ---
    # Using glob is a reliable way to find all files matching a pattern.
    search_pattern = os.path.join(input_dir, '*.hdr')
    envi_files = glob.glob(search_pattern)

    if not envi_files:
        print(f"No ENVI header (.hdr) files found in '{input_dir}'.")
        return

    print(f"\nFound {len(envi_files)} ENVI images to process.")

    # --- 3. Loop through each file and process it ---
    for hdr_path in envi_files:
        classify_and_save_image(hdr_path, output_dir)
    
    print("\n" + "="*80)
    print("--- Batch processing complete for all images. ---")
    print("="*80)


if __name__ == '__main__':
    # --- CONFIGURE YOUR DIRECTORIES HERE ---
    INPUT_DIRECTORY = 'path/to/your/envi_images'
    OUTPUT_DIRECTORY = 'path/to/your/classification_outputs'
    
    # Run the main function
    batch_classify_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY)