# CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation

## Overview
This repository implements a novel knowledge distillation framework for machine translation tasks, featuring both standard and softmax-based iterative distillation approaches. The framework enables efficient transfer of knowledge from large teacher models to smaller student models while maintaining translation quality.

## Key Features
- **Standard Iterative Distillation**: Implements conventional knowledge distillation with LoRA-based parameter-efficient fine-tuning
- **Softmax-based Distillation**: Novel approach utilizing teacher model's probability distributions for enhanced knowledge transfer
- **Parameter-Efficient Training**: Leverages LoRA (Low-Rank Adaptation) for memory-efficient model fine-tuning
- **Comprehensive Logging**: Integration with Weights & Biases for experiment tracking and visualization

## Architecture
The framework consists of three main components:
1. **Data Generation Module**: Extracts teacher model outputs and probability distributions
2. **Standard Distillation**: Implements conventional knowledge distillation with LoRA
3. **Softmax-based Distillation**: Advanced distillation using teacher model's probability distributions

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Unsloth
- Weights & Biases
- CUDA-compatible GPU

## Usage
### Data Generation
```bash
python generate_distillation_data.py \
    --model_name "google/gemma-2-9b" \
    --input_file "path/to/input.txt" \
    --output_dir "path/to/output" \
    --hf_token "your_hf_token" \
    --instruction "Your translation instruction" \
    --k 20 \
    --quantize_bits 8 \
    --max_samples 20000 \
    --max_new_tokens 64 \
    --cache_dir "path/to/cache" \
    --device "cuda"
```

### Standard Distillation
```bash
python iterative_distillation.py \
    --student_model_name "model_name" \
    --output_model_name "output_name" \
    --data_path "path/to/data.json" \
    --hf_token "your_hf_token" \
    --wandb_token "your_wandb_token" \
    --cache_dir "path/to/cache" \
    --wandb_project "your_project" \
    --wandb_run_name "your_run_name" \
    --learning_rate 2e-4 \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 100 \
    --lora_r 2 \
    --lora_alpha 4 \
    --lora_dropout 0 \
    --seed 3407 \
    --prompt_template "your_prompt_template"
```

### Softmax-based Distillation
```bash
python softmax_iterative_distillation.py \
    --student_model_name "model_name" \
    --output_model_name "output_name" \
    --data_path "path/to/data.json" \
    --teacher_logits_dir "path/to/logits" \
    --hf_token "your_hf_token" \
    --wandb_token "your_wandb_token" \
    --cache_dir "path/to/cache" \
    --wandb_project "your_project" \
    --wandb_run_name "your_run_name" \
    --learning_rate 2e-4 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_steps 120 \
    --lora_r 2 \
    --lora_alpha 4 \
    --lora_dropout 0 \
    --seed 3407 \
    --distillation_alpha 0.5 \
    --temperature 1.0 \
    --validation_split 0.2 \
    --max_samples 20000 \
    --prompt_template "your_prompt_template"
```

## Data Format
The framework expects the input data in the following format:
```
source_text###>target_text
```
Each line should contain a source-target pair separated by "###>".

## Output Format
The framework generates the following outputs:
1. **Metadata JSON**: Contains information about the generated data
2. **NumPy Arrays**: Contains probability distributions for each token
3. **Model Checkpoints**: Trained student models

## Citation
If you use this code in your research, please cite:
```
@misc{cycledistill2025,
    title={CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation},
    author={Deepon Halder, Thanmay Jayakumar, Raj Dabre},
    year={2025},
    journal={},
    url={\url{https://github.com/deeps73/CycleDistill}}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
