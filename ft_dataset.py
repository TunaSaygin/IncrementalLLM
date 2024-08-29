import os
import json
import random
from dataset_utils import create_samples, create_easy_hard_contrastive_data
wanted_slots = ["taxi-departure","taxi-destination"]
sample_size = 20
def process_problematic_turns(directory, sample_size=5):
    all_selected_samples = []

    # Traverse the directory structure
    for slot_name in os.listdir(directory):
        slot_dir = os.path.join(directory, slot_name)
        if os.path.isdir(slot_dir) and slot_name in wanted_slots:
            # Load fn.json and fp.json files
            fn_path = os.path.join(slot_dir, 'fn_train_500.json')
            fp_path = os.path.join(slot_dir, 'fp_train_500.json')

            if os.path.exists(fn_path):
                with open(fn_path, 'r') as fn_file:
                    fn_data = json.load(fn_file)
                    selected_fn_samples = create_samples(fn_data, sample_size)
                    all_selected_samples.extend(selected_fn_samples)

            if os.path.exists(fp_path):
                with open(fp_path, 'r') as fp_file:
                    fp_data = json.load(fp_file)
                    selected_fp_samples = create_samples(fp_data, sample_size)
                    all_selected_samples.extend(selected_fp_samples)

    return all_selected_samples

# Example usage
directory = 'problematic_cases'  # Path to your problematic_turns directory
# sample_size = 5  # Number of samples to retrieve from each fn.json and fp.json
selected_samples = process_problematic_turns(directory, sample_size)
#load the correct turns
with open("correct_samples/handpicked_correct_examples/handpicked_correct1.json","r") as f:
    correct_examples = json.load(f)
selected_samples.extend(correct_examples)
random.seed(42)
random.shuffle(selected_samples)
# Save the selected samples if needed
with open('ft_dataset/aug8/SFT_train_25C_80P_taxi.json', 'w') as f:
    json.dump(selected_samples, f, ensure_ascii=False, indent=4)

print(f"Total selected samples: {len(selected_samples)}")