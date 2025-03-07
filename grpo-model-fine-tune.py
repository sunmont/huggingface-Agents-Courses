import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login
import os
from dotenv import load_dotenv

import torch
def supports_flash_attention(device_id):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90

ideal_length = 50
def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]

load_dotenv(override=True)
hf_token = os.getenv('HF_TOKEN')
wabDB_key = os.getenv('WANDB_KEY')

login(hf_token)
# Log to Weights & Biases
wandb.login(key=wabDB_key)

# Load dataset
dataset = load_dataset("mlabonne/smoltldr")
print(dataset)

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    #attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# Training arguments
training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
wandb.init(project="GRPO")
print("Training ...")
trainer.train()