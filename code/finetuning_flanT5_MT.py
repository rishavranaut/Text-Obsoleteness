import nltk
import torch
import numpy as np
from huggingface_hub import HfFolder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score,recall_score,f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict


label2id = {"0": 0, "1": 1}
id2label = {id: label for label, id in label2id.items()}


def format_data_T1(row, flag=-1):
    prompt = f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """'Determine whether a text passage has been updated by identifying changes in date, numbers, scores, statuses, or other relevant information between two given sentences. Provide a binary answer (Yes/No) indicating if the new sentence represents an update to the old sentence.'"""

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

def load_dataset(model_type: str = "") -> Dataset:
    if model_type == "AutoModelForSequenceClassification":
        # Load CSV files
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('val.csv')

        # Initialize empty DataFrames for Task 2
        train_df_T2 = pd.DataFrame(columns=['text', 'Ground-Truth'])
        test_df_T2 = pd.DataFrame(columns=['text', 'Ground-Truth'])

        # Apply formatting for Task 1
        train_df["text"] = train_df.apply(lambda row: format_data_T1(row, flag=0), axis=1)
        test_df["text"] = test_df.apply(lambda row: format_data_T1(row, flag=0), axis=1)

        # Apply formatting for Task 2
        train_df_T2["text"] = train_df.apply(lambda row: format_data_T2(row, flag=0), axis=1)
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

    return dataset


MODEL_ID = "google/flan-t5-large"
REPOSITORY_ID = "13ari/flan_T5_MT"

config = AutoConfig.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir = REPOSITORY_ID,
    learning_rate = 1e-4,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    weight_decay = 0.01,
    logging_strategy="steps",
    logging_steps=2500,
    report_to="tensorboard",
    evaluation_strategy = 'steps',
    save_strategy='steps',
    save_steps=2500, 
    load_best_model_at_end = False,
    save_total_limit=2,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id= REPOSITORY_ID,
    hub_token='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',
    # resume_from_checkpoint=True
)


def tokenize_function(examples) -> dict:
    """Tokenize the text column in the dataset"""
    sentences = [
        text for text in examples['text']
    ]
    return tokenizer(sentences, truncation=True, padding="max_length", max_length=512)

def compute_metrics(eval_pred) -> dict:
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # if the model also returns hidden_states or attentions
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        'Accuracy': accuracy_score(labels, predictions),
        'Precision': precision_score(labels, predictions,average='binary'),
        'Recall': recall_score(labels, predictions,average='binary'),
        'F1 Score': f1_score(labels, predictions,average='binary'),
        # 'Classification Report': classification_report(labels, predictions, output_dict=True)  # output_dict=True ensures a dict is returned
    }
    torch.cuda.empty_cache()
    
    return metrics



def train() -> None:
    """
    Train the model and save it to the Hugging Face Hub.
    """
    dataset = load_dataset("AutoModelForSequenceClassification")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    nltk.download("punkt")

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # SAVE AND EVALUATE
    tokenizer.save_pretrained(REPOSITORY_ID)
    trainer.create_model_card()
    trainer.push_to_hub()
    print(trainer.evaluate())

if __name__ == "__main__":
    train()