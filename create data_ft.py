import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
max_sample_per_file = 20
random.seed(42)
for i in range (11):
  # Load the dataset
  with open('correct_turns_train_500.json', 'r') as f:
      data = json.load(f)

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
      if len(selected_indices) >= max_sample_per_file + 10*i:  # For example, selecting 100 diverse samples
          break

  # Step 4: Retrieve the selected samples
  selected_samples = [data[i] for i in selected_indices]

  # Save or use your selected samples
  with open(f'correct_samples/selected_samples_{42}_{max_sample_per_file + 10*i}.json', 'w') as f:
      json.dump(selected_samples, f, ensure_ascii=False, indent=4)

  print(f"Selected {len(selected_samples)} diverse samples.")
