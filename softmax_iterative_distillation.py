"""
CycleDistill: Knowledge Distillation Framework for Machine Translation
Softmax-based Iterative Distillation Module

This module implements the advanced softmax-based iterative knowledge distillation
component of the CycleDistill framework. It utilizes the teacher model's probability
distributions over vocabulary tokens to guide the student model's learning process.
The implementation incorporates a novel distillation loss function that combines
standard cross-entropy with KL divergence between teacher and student distributions.

Author: Deepon Halder, Thanmay Jayakumar, Raj Dabre
Institution: AI4Bharat
Date: 2025
"""

import os
import json
import torch
import numpy as np
from datasets import Dataset
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.nn import functional as F
import logging
from tqdm import tqdm
import argparse

# Argument parsing for distillation configuration
# Includes model, data, training, and LoRA parameters

def parse_args():
    """
    Parse command line arguments for softmax-based iterative distillation configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - hf_token: Hugging Face API token
            - wandb_token: Weights & Biases API token
            - student_model_name: Name of the student model
            - output_model_name: Name for the output model
            - cache_dir: Directory for model caching
            - data_path: Path to distillation data
            - teacher_logits_dir: Directory containing teacher logits
            - wandb_project: Weights & Biases project name
            - wandb_run_name: Weights & Biases run name
            - learning_rate: Training learning rate
            - epochs: Number of training epochs
            - batch_size: Training batch size
            - gradient_accumulation_steps: Steps for gradient accumulation
            - max_steps: Maximum training steps
            - lora_r: LoRA rank parameter
            - lora_alpha: LoRA alpha parameter
            - lora_dropout: LoRA dropout rate
            - seed: Random seed for reproducibility
            - distillation_alpha: Weight for distillation loss
            - temperature: Temperature for softening distributions
            - validation_split: Validation set split ratio
            - max_samples: Maximum number of samples
            - prompt_template: Instruction prompt template
    """
    parser = argparse.ArgumentParser(description="Softmax-based iterative knowledge distillation for machine translation")
    parser.add_argument("--hf_token", type=str, required=True,
                      help="Hugging Face API token for model access")
    parser.add_argument("--wandb_token", type=str, required=True,
                      help="Weights & Biases API token for experiment tracking")
    parser.add_argument("--student_model_name", type=str, required=True,
                      help="Name of the student model to use for distillation")
    parser.add_argument("--output_model_name", type=str, required=True,
                      help="Name for the output model on Hugging Face Hub")
    parser.add_argument("--cache_dir", type=str, required=True,
                      help="Directory for caching model files")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to the distillation data JSON file")
    parser.add_argument("--teacher_logits_dir", type=str, required=True,
                      help="Directory containing teacher model logits")
    parser.add_argument("--wandb_project", type=str, required=True,
                      help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                      help="Weights & Biases run name")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                      help="Number of steps for gradient accumulation")
    parser.add_argument("--max_steps", type=int, default=120,
                      help="Maximum number of training steps")
    parser.add_argument("--lora_r", type=int, default=2,
                      help="LoRA rank for parameter-efficient fine-tuning")
    parser.add_argument("--lora_alpha", type=int, default=4,
                      help="LoRA alpha parameter for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0,
                      help="LoRA dropout rate")
    parser.add_argument("--seed", type=int, default=3407,
                      help="Random seed for reproducibility")
    parser.add_argument("--distillation_alpha", type=float, default=0.5,
                      help="Weight for distillation loss in the combined loss function")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for softening probability distributions")
    parser.add_argument("--validation_split", type=float, default=0.2,
                      help="Validation set split ratio")
    parser.add_argument("--max_samples", type=int, default=20000,
                      help="Maximum number of samples to process")
    parser.add_argument("--prompt_template", type=str, required=True,
                      help="Template for the instruction prompt")
    return parser.parse_args()

# Custom collator for distillation, handling teacher logits and student input alignment
class DistillationCollator:
    """
    Custom data collator for distillation training.
    
    This collator handles the alignment of teacher logits with student model inputs,
    ensuring proper padding and masking for efficient batch processing.
    
    Attributes:
        tokenizer: Tokenizer instance for text processing
        pad_token_id: ID of the padding token
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the distillation collator.
        
        Args:
            tokenizer: Tokenizer instance for text processing
        """
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        """
        Process a batch of features for distillation training.
        
        Args:
            features (list): List of feature dictionaries containing:
                - example_idx: Index of the example
                - input_ids: Token IDs for input text
                - labels: Token IDs for target text
        
        Returns:
            dict: Processed batch containing:
                - input_ids: Padded input token IDs
                - attention_mask: Attention mask for padding
                - labels: Padded target token IDs
                - example_idx: Example indices for teacher logits lookup
        """
        example_indices = [f.pop("example_idx") for f in features]
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_mask = []
        padded_labels = []
        
        for ids, labs in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            padded_lab = labs + [-100] * padding_length
            padded_input_ids.append(padded_ids)
            attention_mask.append(mask)
            padded_labels.append(padded_lab)
        
        batch = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(padded_labels),
            "example_idx": torch.tensor(example_indices)
        }
        
        return batch

# Custom Trainer for softmax-based distillation, combining standard and KL loss
class DistillationTrainer(Trainer):
    """
    Custom trainer for softmax-based distillation.
    
    This trainer implements a novel distillation loss function that combines:
    1. Standard cross-entropy loss for sequence generation
    2. KL divergence loss between teacher and student distributions
    
    The implementation supports:
    - Temperature-scaled softmax for distribution softening
    - Weighted combination of standard and distillation losses
    - Efficient handling of sparse teacher distributions
    """
    
    def __init__(self, teacher_logits_map, alpha=0.5, temperature=1.0, **kwargs):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_logits_map (dict): Mapping of example indices to teacher logits
            alpha (float): Weight for distillation loss in combined loss
            temperature (float): Temperature for softening distributions
            **kwargs: Additional arguments for the base Trainer class
        """
        super().__init__(**kwargs)
        self.teacher_logits_map = teacher_logits_map
        self.alpha = alpha
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute the combined loss for distillation training.
        
        Args:
            model: Student model instance
            inputs (dict): Input tensors for the model
            return_outputs (bool): Whether to return model outputs
            num_items_in_batch (int): Number of items in the current batch
            **kwargs: Additional arguments for loss computation
        
        Returns:
            tuple or float: Combined loss and optional model outputs
        """
        example_indices = inputs.pop("example_idx", None)
        outputs = model(**inputs)
        standard_loss = outputs.loss
        
        if example_indices is None:
            return (standard_loss, outputs) if return_outputs else standard_loss
        
        student_logits = outputs.logits
        distillation_loss = self.compute_distillation_loss(student_logits, example_indices, inputs)
        
        if distillation_loss is not None:
            total_loss = (1 - self.alpha) * standard_loss + self.alpha * distillation_loss
        else:
            total_loss = standard_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_distillation_loss(self, student_logits, example_indices, inputs):
        """
        Compute the distillation loss using teacher model distributions.
        
        Args:
            student_logits (torch.Tensor): Logits from student model
            example_indices (torch.Tensor): Indices for teacher logits lookup
            inputs (dict): Input tensors for the model
        
        Returns:
            torch.Tensor or None: Computed distillation loss or None if unavailable
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            return None
        
        distill_loss = 0.0
        count = 0
        
        response_marker = "### Response: \n"
        response_tokens = self.tokenizer.encode(response_marker, add_special_tokens=False)
        
        for b, idx in enumerate(example_indices.tolist()):
            if idx not in self.teacher_logits_map:
                continue
            
            teacher_data = self.teacher_logits_map[idx]
            teacher_indices = torch.tensor(teacher_data['indices']).to(student_logits.device)
            teacher_values = torch.tensor(teacher_data['values']).to(student_logits.device)
            
            ids = input_ids[b].tolist()
            response_start_pos = -1
            
            for i in range(len(ids) - len(response_tokens) + 1):
                if ids[i:i+len(response_tokens)] == response_tokens:
                    response_start_pos = i + len(response_tokens)
                    break
            
            if response_start_pos == -1:
                continue
            
            positions = min(seq_len, len(teacher_indices))
            for pos in range(positions):
                if pos < response_start_pos:
                    continue
                
                pos_student_logits = student_logits[b, pos]
                teacher_sparse = torch.zeros(vocab_size, device=student_logits.device)
                pos_indices = teacher_indices[pos]
                pos_values = teacher_values[pos]
                
                for i, val in enumerate(pos_values):
                    token_idx = int(pos_indices[i])
                    if 0 <= token_idx < vocab_size:
                        teacher_sparse[token_idx] = val
                
                if teacher_sparse.sum() > 0:
                    teacher_sparse = teacher_sparse / teacher_sparse.sum()
                else:
                    continue
                
                soft_student_logits = pos_student_logits / self.temperature
                student_log_probs = F.log_softmax(soft_student_logits, dim=0)
                pos_loss = F.kl_div(
                    student_log_probs,
                    teacher_sparse,
                    reduction='sum'
                )
                
                distill_loss += pos_loss
                count += 1
        
        if count > 0:
            return distill_loss / count
        return None

def main():
    """
    Main execution function for softmax-based iterative distillation.
    
    This function orchestrates the advanced distillation process:
    1. Loads and processes teacher model logits
    2. Initializes student model with LoRA adaptation
    3. Prepares training and validation datasets
    4. Configures the distillation training environment
    5. Executes the distillation process with combined loss
    6. Saves the distilled model
    
    The implementation features:
    - Novel softmax-based distillation approach
    - Efficient handling of teacher distributions
    - Comprehensive experiment tracking
    - Validation-based training monitoring
    """
    # Set up environment and logging
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    os.environ["HUGGINGFACE_TOKEN"] = args.hf_token

    print(f"Total CUDA : {torch.cuda.device_count()}")
    print(f"Current CUDA : {torch.cuda.current_device()}")

    # Load teacher logits for all examples (sparse top-k per token)
    teacher_logits_map = {}
    print("Loading teacher model logits files...")
    for i in tqdm(range(args.max_samples)):
        indices_path = os.path.join(args.teacher_logits_dir, f"topk_indices_{i}.npy")
        values_path = os.path.join(args.teacher_logits_dir, f"topk_values_{i}.npy")
        
        if os.path.exists(indices_path) and os.path.exists(values_path):
            try:
                teacher_logits_map[i] = {
                    'indices': np.load(indices_path),
                    'values': np.load(values_path)
                }
            except Exception as e:
                logger.warning(f"Could not load logits for example {i}: {e}")

    logger.info(f"Loaded teacher logits for {len(teacher_logits_map)} examples")

    # Load student model and tokenizer with LoRA support
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=False,
        cache_dir=args.cache_dir
    )

    model = model.to("cuda")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load distillation data (teacher outputs and references)
    with open(args.data_path, "r", encoding="utf-8") as f:
        distillation_data_json = json.load(f)
    distillation_sequences = distillation_data_json.get("sequences", [])

    alpaca_prompt = args.prompt_template

    # Prepare dataset for supervised fine-tuning
    def create_dataset():
        inputs = ["Hindi Text : "+seq["input_text"]+"\n English Text: \n\n ### Response: \n"+seq["output_text"] for seq in distillation_sequences]
        outputs = [seq["output_text"] for seq in distillation_sequences]
        texts = []
        example_indices = []
        
        EOS_TOKEN = tokenizer.eos_token
        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            text = alpaca_prompt.format(input) + EOS_TOKEN
            texts.append(text)
            example_indices.append(idx)
        
        return {
            "text": texts,
            "example_idx": example_indices
        }

    raw_dataset = Dataset.from_dict(create_dataset())
    print(raw_dataset[5])
    logger.info(f"Loaded {len(raw_dataset)} examples from distillation dataset")

    # Tokenize dataset for model input
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=min(tokenizer.model_max_length, 2048),
            return_tensors=None
        )
        
        tokenized["example_idx"] = examples["example_idx"]
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=["text"]
    )

    tokenized_dataset = tokenized_dataset.shuffle(seed=args.seed)
    train_size = int(len(tokenized_dataset) * (1 - args.validation_split))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    logger.info(f"Training on {len(train_dataset)} examples")
    logger.info(f"Validating on {len(val_dataset)} examples")

    # Initialize Weights & Biases for experiment tracking
    wandb.login(key=args.wandb_token)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.student_model_name,
            "output_model": args.output_model_name,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "validation_split": args.validation_split,
            "distillation_alpha": args.distillation_alpha,
            "temperature": args.temperature
        }
    )

    data_collator = DistillationCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=1,
        fp16=True,
        bf16=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=5,
        max_steps=args.max_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=args.output_model_name,
        hub_token=args.hf_token,
        remove_unused_columns=False
    )

    # Trainer for softmax-based distillation
    trainer = DistillationTrainer(
        teacher_logits_map=teacher_logits_map,
        alpha=args.distillation_alpha,
        temperature=args.temperature,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    print("Training Start")
    torch.cuda.empty_cache()
    trainer.train()
    print("Training End")

    print("Pushing to Hub")
    model.push_to_hub(args.output_model_name, token=args.hf_token, cache_dir=args.cache_dir)
    tokenizer.push_to_hub(args.output_model_name, token=args.hf_token, cache_dir=args.cache_dir)

    wandb.finish()
    print(f"Model successfully trained and pushed to {args.output_model_name}")

if __name__ == "__main__":
    main()
