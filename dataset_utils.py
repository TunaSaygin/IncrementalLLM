import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def create_samples(data,sample_size):
    random.seed(42)

    # For simplicity, let's assume you're using the dialogue history and slot values for vectorization
    def prepare_text_for_vectorization(sample):
        history = ' '.join(sample['utt'])  # Convert history to a single string
        slots = ' '.join([f"{k}:{v}" for k, v in sample['bs_gt'].items()])  # Convert slots to a single string
        return history + ' ' + slots

    # Create a list of texts from your samples
    texts = [prepare_text_for_vectorization(sample) for sample in data]

    # Step 1: Vectorize the dialogue samples using TF-IDF (or other embeddings)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Step 2: Calculate cosine similarity between all samples
    similarity_matrix = cosine_similarity(X)

    # Step 3: Select diverse samples based on cosine similarity
    selected_indices = []
    remaining_indices = list(range(len(data)))

    # Start with a random sample
    selected_indices.append(random.choice(remaining_indices))
    remaining_indices.remove(selected_indices[-1])

    # Iteratively select samples with the lowest similarity to the selected set
    while remaining_indices:
        last_selected_index = selected_indices[-1]
        similarities = similarity_matrix[last_selected_index, remaining_indices]

        # Find the index with the lowest similarity
        min_similarity_index = remaining_indices[np.argmin(similarities)]

        # Add this index to the selected set
        selected_indices.append(min_similarity_index)
        remaining_indices.remove(min_similarity_index)

        # Optional: Stop when you have enough samples
        if len(selected_indices) >= sample_size:  # For example, selecting 100 diverse samples
            break

    # Step 4: Retrieve the selected samples
    selected_samples = [data[i] for i in selected_indices]

    return selected_samples


# Function to calculate slot differences
def calculate_slot_differences(gt_slots, pred_slots):
    diff_count = 0
    for slot in gt_slots:
        if slot not in pred_slots or gt_slots[slot] != pred_slots[slot]:
            diff_count += 1
    for slot in pred_slots:
        if slot not in gt_slots:
            diff_count += 1
    return diff_count

#
def create_easy_hard_contrastive_data(data):
    easy_examples = []
    hard_examples = []
    threshold = 2  # You can adjust this threshold based on your data

    for example in data:
        gt_slots = example['bs_gt']
        pred_slots = json.loads(example['bs_pred'])  # Assuming `bs_pred` is a JSON string
        diff_count = calculate_slot_differences(gt_slots, pred_slots)
        
        if diff_count <= threshold:
            easy_examples.append(example)
        else:
            hard_examples.append(example)

    # Balance the dataset
    num_samples = 100
    easy_samples = random.sample(easy_examples, num_samples // 2)
    hard_samples = random.sample(hard_examples, num_samples // 2)

    # Combine the balanced set
    balanced_dataset = easy_samples + hard_samples
    random.shuffle(balanced_dataset)

    return balanced_dataset