"""
CycleDistill: Knowledge Distillation Framework for Machine Translation
Data Generation Module

This module implements the data generation component of the CycleDistill framework,
which extracts teacher model outputs and probability distributions for subsequent
knowledge distillation. The module employs a large language model (LLM) as the
teacher model to generate high-quality translations and their associated probability
distributions over the vocabulary.

Author: Deepon Halder, Thanmay Jayakumar, Raj Dabre
Institution: AI4Bharat
Date: 2025
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import re
import json
import os
import argparse

def parse_args():
    """
    Parse command line arguments for data generation configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - model_name: Name of the teacher model
            - k: Number of top probabilities to store
            - quantize_bits: Bit precision for probability quantization
            - output_dir: Directory for saving generated data
            - input_file: Path to input text file
            - hf_token: Hugging Face API token
            - max_samples: Maximum number of samples to process
            - max_new_tokens: Maximum sequence length for generation
            - instruction: Task-specific instruction prompt
            - cache_dir: Directory for caching model files
            - device: Device to run the model on (cuda/cpu)
    """
    parser = argparse.ArgumentParser(description="Generate distillation data using a teacher model")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the teacher model to use for generating translations")
    parser.add_argument("--k", type=int, default=20,
                      help="Number of top probabilities to store for each token")
    parser.add_argument("--quantize_bits", type=int, default=8, choices=[8, 16],
                      help="Number of bits for probability quantization (8 or 16)")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the generated distillation data")
    parser.add_argument("--input_file", type=str, required=True,
                      help="Path to the input text file containing source-target pairs")
    parser.add_argument("--language_name", type=str, required=True,
                      help="Name of the language to be translated")
    parser.add_argument("--hf_token", type=str, required=True,
                      help="Hugging Face API token for model access")
    parser.add_argument("--max_samples", type=int, default=20000,
                      help="Maximum number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                      help="Maximum number of new tokens to generate per translation")
    parser.add_argument("--instruction", type=str, required=True,
                      help="Instruction prompt for the translation task")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Directory for caching model files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run the model on (cuda/cpu)")
    return parser.parse_args()

def create_alpaca_prompt(source_text, instruction,language_name):
    """
    Create a formatted prompt for the translation task using the Alpaca template.
    
    Args:
        source_text (str): Source text to be translated
        instruction (str): Task-specific instruction
        
    Returns:
        str: Formatted prompt following the Alpaca template structure
    """
    return f"""### Instruction: {instruction}
### Input: {language_name} Text: {source_text} 
### Response:"""

def extract_translation(text):
    """
    Extract translation from generated text using regex pattern matching.
    
    Args:
        text (str): Generated text containing the translation
        
    Returns:
        str: Extracted translation or original text if pattern not found
    """
    if isinstance(text, str):
        match = re.search(r'### Response: (.*)', text, re.DOTALL)
        return match.group(1).strip() if match else text
    return text

def main():
    """
    Main execution function for data generation.
    
    This function orchestrates the data generation process:
    1. Initializes the teacher model and tokenizer
    2. Processes input data
    3. Generates translations and probability distributions
    4. Saves the generated data in a structured format
    
    The generated data includes:
    - Source-target text pairs
    - Token-level probability distributions
    - Metadata for subsequent distillation
    """
    args = parse_args()
    os.environ["HUGGINGFACE_TOKEN"] = args.hf_token
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "numpy_data"), exist_ok=True)

    # Initialize teacher model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        token=args.hf_token,
        cache_dir=args.cache_dir
    )
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    model.eval()

    # Load dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        translations = [line.strip() for line in f if line.strip()]

    # Initialize metadata structure
    metadata = {
        "model_name": args.model_name,
        "top_k": args.k,
        "quantize_bits": args.quantize_bits,
        "device": args.device,
        "sequences": []
    }

    # Process translations
    for idx, translation in enumerate(tqdm(translations[:args.max_samples])):
        try:
            # Prepare input
            source_text, target_text = translation.split("###>")
            prompt = create_alpaca_prompt(source_text.strip(), args.instruction,args.language_name)
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            
            # Process generated output
            decoded_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            formatted_output = extract_translation(decoded_output)
            
            # Process and store probability distributions
            all_scores = outputs.scores
            topk_values = []
            topk_indices = []
            
            for score in all_scores:
                probs = torch.softmax(score[0], dim=-1).cpu().numpy()
                indices = np.argpartition(probs, -args.k)[-args.k:]
                values = probs[indices]
                
                sorted_idx = np.argsort(values)[::-1]
                values = values[sorted_idx]
                indices = indices[sorted_idx]
                
                if args.quantize_bits == 8:
                    values = (values * 255).astype(np.uint8)
                elif args.quantize_bits == 16:
                    values = values.astype(np.float16)
                
                topk_values.append(values)
                topk_indices.append(indices.astype(np.uint16))
            
            if len(topk_values) > 0:
                # Store sequence data
                formatted_output_ids = tokenizer(formatted_output, return_tensors="pt").input_ids[0]
                formatted_output_subwords = tokenizer.convert_ids_to_tokens(formatted_output_ids)
                
                sequence_data = {
                    "id": idx,
                    "input_text": source_text.strip(),
                    "output_text": formatted_output,
                    "subwords": formatted_output_subwords,
                }
                metadata["sequences"].append(sequence_data)
                
                # Save probability distributions
                np.save(
                    os.path.join(args.output_dir, "numpy_data", f"topk_values_{idx}.npy"),
                    np.vstack(topk_values)
                )
                np.save(
                    os.path.join(args.output_dir, "numpy_data", f"topk_indices_{idx}.npy"),
                    np.vstack(topk_indices)
                )
                
                # Periodically save metadata
                if (idx + 1) % 100 == 0:
                    with open(os.path.join(args.output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
            else:
                print(f"Warning: No logits captured for sequence {idx}")
                
        except Exception as e:
            print(f"Error processing sequence {idx}: {str(e)}")
            continue

    # Save final metadata
    with open(os.path.join(args.output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
