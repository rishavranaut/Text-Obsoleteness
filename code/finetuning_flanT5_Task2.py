import nltk
import json
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
import pandas as pd
from datasets import Dataset, DatasetDict

label2id = {"0": 0, "1": 1}
id2label = {id: label for label, id in label2id.items()}


def create_jsonl_entry_test(row):
    prompt = (
        f"Analyze the given text based on its contextual understanding to determine whether any factual updates"
        f"(e.g., date changes, numerical updates, score modifications, or status changes) are likely to occur in the future."
        f"Return a response indicating 'Yes' if an update is predicted and 'No' otherwise."
        f"Text: {row['old_cleaned_content']} +\nAnswer:"
    )
    return json.dumps({"prompt": prompt, "completion": ''})


# Apply the function to each row and store the result in a new column 'text'

# train

def load_dataset(model_type: str = "") -> Dataset:
    if model_type == "AutoModelForSequenceClassification":
        train_df = pd.read_csv("train.csv", encoding='ISO-8859-1')
        train_df['text'] = train_df.apply(create_jsonl_entry_test, axis=1)
        train_df.drop(columns=['old_cleaned_content','new_cleaned_content'], inplace=True)

        # validation
        test_df = pd.read_csv("val.csv", encoding='ISO-8859-1')
        test_df['text'] = test_df.apply(create_jsonl_entry_test, axis=1)
        test_df.drop(columns=['old_cleaned_content','new_cleaned_content'], inplace=True)

        train_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)
        test_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True) 

        dataset_train = Dataset.from_pandas(train_df)
        dataset_test = Dataset.from_pandas(test_df)

        dataset = DatasetDict({
            'train': dataset_train,
            'test': dataset_test
        })  

    return dataset

MODEL_ID = "google/flan-t5-large"
REPOSITORY_ID = "13ari/flanT5_Task2"

config = AutoConfig.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

training_args = TrainingArguments(
    num_train_epochs=15,
    output_dir = REPOSITORY_ID,
    learning_rate = 1e-4,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
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
    resume_from_checkpoint=True
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