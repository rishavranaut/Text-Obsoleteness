import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
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

# preparing the data
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


login('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')


# The model that you want to train from the Hugging Face hub
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name='meta-llama/Meta-Llama-3-8B'

# The instruction dataset to use

# Fine-tuned model name
new_model = "llama3_MT"

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
num_train_epochs = 5

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


# Shuffle the dataframes to ensure randomness
train_df = pd.read_csv("train.csv", encoding='ISO-8859-1')
test_df = pd.read_csv("val.csv", encoding='ISO-8859-1')

# Initialize empty DataFrames for Task 2
train_df_T2 = pd.DataFrame(columns=['text', 'Ground-Truth'])
test_df_T2 = pd.DataFrame(columns=['text', 'Ground-Truth'])

# Function to format data for Task 1
def format_data_T1(row, flag=-1):
    prompt = f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """Determine whether a text passage has been updated by identifying changes in date, 
    numbers, scores, statuses, or other relevant information between two given sentences. 
    Provide a binary answer (Yes/No) indicating if the new sentence represents an update to the old sentence.\n"""

    formatted_entry = {"prompt": base_prompt + prompt, "completion": completion} if flag == 1 else {"prompt": base_prompt + prompt, "completion": ''}
    return json.dumps(formatted_entry)

# Function to format data for Task 2
def format_data_T2(row, flag=-1):
    prompt = f"Text: {row['old_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """Analyze the given text based on its contextual understanding to determine whether any factual updates (e.g., date changes, numerical updates, score modifications, or status changes) are likely to occur in the future. 
    Return a response indicating "Yes" if an update is predicted and "No" otherwise.\n"""

    formatted_entry = {"prompt": base_prompt + prompt, "completion": completion} if flag == 1 else {"prompt": base_prompt + prompt, "completion": ''}
    return json.dumps(formatted_entry)

# Apply formatting for Task 1
train_df["text"] = train_df.apply(lambda row: format_data_T1(row, flag=1), axis=1)
test_df["text"] = test_df.apply(lambda row: format_data_T1(row, flag=0), axis=1)



# Apply formatting for Task 2
train_df_T2["text"] = train_df.apply(lambda row: format_data_T2(row, flag=1), axis=1)
test_df_T2["text"] = test_df.apply(lambda row: format_data_T2(row, flag=0), axis=1)


# Keep only necessary columns
train_df = train_df[['text', 'Ground-Truth']]
test_df = test_df[['text', 'Ground-Truth']]


# Ensure labels column consistency before merging
train_df_T2["Ground-Truth"] = train_df["Ground-Truth"].values
test_df_T2["Ground-Truth"] = test_df["Ground-Truth"].values

# Merge both tasks' datasets
train_df = pd.concat([train_df, train_df_T2], ignore_index=True)
test_df = pd.concat([test_df, test_df_T2], ignore_index=True)

# Rename column for Hugging Face dataset compatibility
train_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)
test_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)

# Shuffle datasets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert DataFrames to Hugging Face Dataset objects
dataset_train = Dataset.from_pandas(train_df)
dataset_test = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'test': dataset_test
})

dataset


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'

)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    # dataset_text_field="text",
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
    peft_config=peft_config,
    # dataset_text_field="text",
    # max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=sft_config,
    # packing=packing,
)

trainer.train()

trainer.model.save_pretrained(new_model)
torch.cuda.empty_cache()
#training complete







# Reload model in FP16 and merge it with LoRA weights

model_name='meta-llama/Meta-Llama-3-8B'
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
new_model='llama3_MT'

model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model



# Merge LoRA weights into the base model (optional, depending on your use case)
model = model.merge_and_unload()

# Save the merged model to a directory
save_directory = "./llama3_MT_merged"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



save_directory = "./llama3_MT_merged"
# Push the saved model directory to the Hub
AutoModelForCausalLM.from_pretrained(save_directory).push_to_hub("13ari/llama3_MT")
AutoTokenizer.from_pretrained(save_directory).push_to_hub("13ari/llama3_MT")
