import os
import json
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from huggingface_hub import login, HfApi
from datasets import DatasetDict, Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import pandas as pd


login('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')


# The model that you want to train from the Hugging Face hub
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name='google/flan-t5-large'

# The instruction dataset to use

# Fine-tuned model name
new_model = "flanT5_Task1"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 8

# Dropout probability for LoRA layers
lora_dropout = 0.05

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 15

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 1e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 200

# Log every X updates steps
logging_steps = 200

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 1024

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


def create_jsonl_entry_train(row):
    prompt = (
        f"Determine whether a text passage has been updated by identifying changes in date, numbers, scores, statuses, "
        f"or other relevant information between two given sentences. Provide a binary answer (Yes/No) indicating if the "
        f"new sentence represents an update to the old sentence.\n"
        f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer: "
    )
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"
    return json.dumps({"prompt": prompt, "completion": completion})

def create_jsonl_entry_test(row):
    prompt = (
        f"Determine whether a text passage has been updated by identifying changes in date, numbers, scores, statuses, "
        f"or other relevant information between two given sentences. Provide a binary answer (Yes/No) indicating if the "
        f"new sentence represents an update to the old sentence.\n"
        f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer: "
    )
    return json.dumps({"prompt": prompt, "completion": ''})


# Apply the function to each row and store the result in a new column 'text'

# train
train_df = pd.read_csv("train.csv", encoding='ISO-8859-1')

train_df['text'] = train_df.apply(create_jsonl_entry_train, axis=1)
train_df.drop(columns=['old_cleaned_content','new_cleaned_content'], inplace=True)

# validation
test_df = pd.read_csv("val.csv", encoding='ISO-8859-1')
test_df['text'] = test_df.apply(create_jsonl_entry_test, axis=1)
test_df.drop(columns=['old_cleaned_content','new_cleaned_content'], inplace=True)


train_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)
test_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)    




# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(train_df)
# dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    # 'val': dataset_val,
    'test': dataset_test
})
dataset





# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map='auto'

)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=min(tokenizer.model_max_length, 1024),
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy='steps',
    save_total_limit=2,
    save_steps=2500,
    logging_strategy="steps",
    logging_steps=2500,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    evaluation_strategy = 'steps',
    load_best_model_at_end = True,
    resume_from_checkpoint=True
)



# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    args=sft_config,
    # packing=packing,
)

trainer.train()

trainer.model.save_pretrained(new_model)
torch.cuda.empty_cache()

#training complete



# saving the model hehehhehehe..

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "flanT5_Task1"
repo_name = "13ari/flanT5_Task1"  # Ensure this repo exists on HF Hub

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')

# Save the model and tokenizer before pushing
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Push to the Hub
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
