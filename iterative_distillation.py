"""
CycleDistill: Knowledge Distillation Framework for Machine Translation
Standard Iterative Distillation Module

This module implements the standard iterative knowledge distillation component of the
CycleDistill framework. It employs LoRA-based parameter-efficient fine-tuning to
transfer knowledge from a teacher model to a smaller student model while maintaining
translation quality. The implementation focuses on efficient memory usage and
scalable training through gradient accumulation.

Author: Deepon Halder, Thanmay Jayakumar, Raj Dabre
Institution: AI4Bharat
Date: 2025
"""

import os
import json
import torch
from datasets import Dataset
import wandb
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import logging
import argparse

# Argument parsing for distillation configuration
# Includes model, data, training, and LoRA parameters

def parse_args():
    """
    Parse command line arguments for standard iterative distillation configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - hf_token: Hugging Face API token
            - wandb_token: Weights & Biases API token
            - student_model_name: Name of the student model
            - output_model_name: Name for the output model
            - cache_dir: Directory for model caching
            - data_path: Path to distillation data
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
            - prompt_template: Instruction prompt template
    """
    parser = argparse.ArgumentParser(description="Standard iterative knowledge distillation for machine translation")
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
    parser.add_argument("--wandb_project", type=str, required=True,
                      help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                      help="Weights & Biases run name")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="Number of steps for gradient accumulation")
    parser.add_argument("--max_steps", type=int, default=100,
                      help="Maximum number of training steps")
    parser.add_argument("--lora_r", type=int, default=2,
                      help="LoRA rank for parameter-efficient fine-tuning")
    parser.add_argument("--lora_alpha", type=int, default=4,
                      help="LoRA alpha parameter for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0,
                      help="LoRA dropout rate")
    parser.add_argument("--seed", type=int, default=3407,
                      help="Random seed for reproducibility")
    parser.add_argument("--prompt_template", type=str, required=True,
                      help="Template for the instruction prompt")
    return parser.parse_args()

def main():
    """
    Main execution function for standard iterative distillation.
    
    This function orchestrates the distillation process:
    1. Initializes the student model with LoRA adaptation
    2. Prepares the training dataset
    3. Configures the training environment
    4. Executes the distillation process
    5. Saves the distilled model
    
    The implementation employs:
    - Parameter-efficient fine-tuning through LoRA
    - Gradient accumulation for memory efficiency
    - Comprehensive experiment tracking via Weights & Biases
    """
    # Set up environment and logging
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    os.environ["HUGGINGFACE_TOKEN"] = args.hf_token

    # Print CUDA memory stats for reproducibility and debugging
    print(f"Total CUDA : {torch.cuda.device_count()}")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.6f}GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024/1024/1024:.6f}GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved()/1024/1024/1024:.6f}GB")

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

    # Apply LoRA adaptation for parameter-efficient fine-tuning
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
            "max_steps": args.max_steps
        }
    )

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load distillation data (teacher outputs and references)
    with open(args.data_path, "r", encoding="utf-8") as f:
        distillation_data_json = json.load(f)

    distillation_sequences = distillation_data_json.get("sequences", [])

    alpaca_prompt = args.prompt_template

    # Prepare dataset for supervised fine-tuning
    def prc():
        inputs = ["Bengali Text : "+seq["input_text"]+"\n English Text: \n\n ### Response: \n"+seq["output_text"] for seq in distillation_sequences]
        texts = []
        EOS_TOKEN = tokenizer.eos_token
        for input in inputs:
            text = alpaca_prompt.format(input) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_dict(prc())
    print(dataset[5])
    logger.info(f"Loaded {len(dataset)} examples from distillation dataset")

    max_seq_length = min(tokenizer.model_max_length, 2048)
    logger.info(f"Using maximum sequence length: {max_seq_length}")

    print(f"Allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.6f}GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024/1024/1024:.6f}GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved()/1024/1024/1024:.6f}GB")

    print(dataset[5])

    # Set up supervised fine-tuning trainer (TRL SFTTrainer)
    from trl import SFTTrainer
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
    )

    print(f"Allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.6f}GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024/1024/1024:.6f}GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved()/1024/1024/1024:.6f}GB")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
    )

    print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    print("Training Start")
    torch.cuda.empty_cache()
    trainer.train()
    print("Training End")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.6f}GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024/1024/1024:.6f}GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved()/1024/1024/1024:.6f}GB")

    print("Pushing to Hub")
    model.push_to_hub(args.output_model_name, token=args.hf_token, cache_dir=args.cache_dir)
    tokenizer.push_to_hub(args.output_model_name, token=args.hf_token, cache_dir=args.cache_dir)

    wandb.finish()
    print(f"Model successfully trained and pushed to {args.output_model_name}")

if __name__ == "__main__":
    main()

