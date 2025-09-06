import numpy as np
import fasttext
import nltk
from typing import List
import os
import urllib.request
import zipfile
import json
import re

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
MODEL_DIR = "/graphics/scratch2/students/webereli/evaluation/prompt_coherence"
JSON_FILE_PATH = "/home/webereli/ViPE-Videos/mp3/apt_ctx_1_sample_True_vipe_True_abst_0_with_characters.json"
UNIQUE_IDENTIFIER = "Alex"
OUTPUT_RESULTS_PATH = "story_coherence_results.json"

os.makedirs(MODEL_DIR, exist_ok=True)
nltk.data.path.append(MODEL_DIR)
nltk.download("punkt", download_dir=MODEL_DIR)
nltk.download("punkt_tab", download_dir=MODEL_DIR)


def remove_unique_identifier(text: str, unique_identifier: str) -> str:
    """
    Remove all occurrences of the unique identifier from the text.
    
    Args:
        text: Input text to clean
        unique_identifier: The identifier to remove (case-insensitive)
        
    Returns:
        Cleaned text with the identifier removed
    """
    if not unique_identifier:
        return text
    
    # Create a pattern that matches the identifier as a whole word (case-insensitive)
    pattern = r'\b' + re.escape(unique_identifier) + r'\b'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces that might be left behind
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def format_prompts_from_json(json_file_path: str) -> tuple[str, int, list]:
    """
    Load prompts from JSON file and format them as a single text for coherence measurement.
    
    Args:
        json_file_path: Path to the JSON file containing prompts
        
    Returns:
        A tuple containing:
        - A single string with all prompts concatenated, treating each prompt as a sentence
        - The number of prompts processed
        - List of individual prompts
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract prompts from the JSON data
    prompts = []
    for item in data:
        if 'prompt' in item and item['prompt']:
            # Clean up the prompt text (remove leading/trailing whitespace)
            prompt = item['prompt'].strip()
            
            # Remove unique identifier if it's not empty
            if UNIQUE_IDENTIFIER:
                prompt = remove_unique_identifier(prompt, UNIQUE_IDENTIFIER)
            
            # Ensure the prompt ends with a period if it doesn't already end with punctuation
            if prompt and not prompt[-1] in '.!?':
                prompt += '.'
            prompts.append(prompt)
    
    # Join all prompts into a single text
    formatted_text = ' '.join(prompts)
    return formatted_text, len(prompts), prompts


def save_coherence_results(similarities: List[float], mean_coherence: float, prompts: List[str], output_path: str):
    """
    Save coherence evaluation results to a JSON file.
    
    Args:
        similarities: List of cosine similarities between consecutive sentences
        mean_coherence: Mean coherence score
        prompts: List of individual prompts
        output_path: Path to save the results JSON file
    """
    results = {
        "evaluation_type": "story_coherence",
        "total_prompts": len(prompts),
        "total_similarities": len(similarities),
        "mean_coherence": float(mean_coherence) if not np.isnan(mean_coherence) else None,
        "coherence_similarities": [float(sim) if not np.isnan(sim) else None for sim in similarities],
        "prompts": prompts,
        "statistics": {
            "min_similarity": float(np.nanmin(similarities)) if len(similarities) > 0 and not np.isnan(np.nanmin(similarities)) else None,
            "max_similarity": float(np.nanmax(similarities)) if len(similarities) > 0 and not np.isnan(np.nanmax(similarities)) else None,
            "std_similarity": float(np.nanstd(similarities)) if len(similarities) > 0 and not np.isnan(np.nanstd(similarities)) else None
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


class FirstOrderCoherence:
    def __init__(self, model_dir: str = "./fasttext_models"):
        """
        Initialize coherence evaluator.

        Args:
            model_dir: Directory where the fastText model will be stored.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_bin_path = os.path.join(model_dir, "crawl-300d-2M-subword.bin")
        zip_path = os.path.join(model_dir, "crawl-300d-2M-subword.zip")

        # Download and extract model if not available
        if not os.path.exists(model_bin_path):
            if not os.path.exists(zip_path):
                print("Downloading fastText model... This may take a while (~4GB).")
                urllib.request.urlretrieve(FASTTEXT_URL, zip_path)
                print("Download complete.")
            else:
                print("Found cached zip file, skipping download.")

            print("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            print("Extraction complete.")
        else:
            print("Model already available, skipping download and extraction.")

        self.model = fasttext.load_model(model_bin_path)

    def sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get the average fastText embedding for a sentence."""
        return self.model.get_sentence_vector(sentence)

    def first_order_coherence(self, text: str) -> List[float]:
        """
        Compute first-order coherence (cosine similarity between consecutive sentences).

        Args:
            text: Input text.

        Returns:
            List of cosine similarities between consecutive sentence embeddings.
        """
        sentences = nltk.sent_tokenize(text, language="english")
        if len(sentences) < 2:
            return [np.nan]

        embeddings = [self.sentence_embedding(sent) for sent in sentences]

        similarities = []
        for i in range(len(embeddings) - 1):
            v1, v2 = embeddings[i], embeddings[i + 1]
            sim = self.cosine_similarity(v1, v2)
            similarities.append(sim)
        return similarities

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return np.nan
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


if __name__ == "__main__":
    # Define output path for results
    
    formatted_prompts, num_prompts, individual_prompts = format_prompts_from_json(JSON_FILE_PATH)
    coherence = FirstOrderCoherence(model_dir=MODEL_DIR)
    scores = coherence.first_order_coherence(formatted_prompts)
    mean_score = np.nanmean(scores)
    
    print("First-order coherence values:", scores)
    print("Mean coherence:", mean_score)
    print(f"Processed {num_prompts} prompts")
    
    # Save results to JSON file
    save_coherence_results(scores, mean_score, individual_prompts, OUTPUT_RESULTS_PATH)