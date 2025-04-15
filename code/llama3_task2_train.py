import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from huggingface_hub import login
login('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')


# preparing the data
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
import pandas as pd
from datasets import DatasetDict, Dataset
import torch 
import gc


def format_data_T1(row, flag=-1):
    prompt = f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """'Determine whether a text passage has been updated by identifying changes in date, numbers, scores, statuses, or other relevant information between two given sentences. Provide a binary answer (Yes/No) indicating if the new sentence represents an update to the old sentence.'"""

    return {"text":str(base_prompt +"\n"+ prompt),"labels": row['Ground-Truth'], "task_ids": 1} 

# Function to format data for Task 2
def format_data_T2(row, flag=-1):
    prompt = f"Text: {row['old_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """Analyze the given text based on its contextual understanding to determine whether any factual updates (e.g., date changes, numerical updates, score modifications, or status changes) are likely to occur in the future. 
    Return a response indicating "Yes" if an update is predicted and "No" otherwise.\n"""

    return {"text":str(base_prompt +"\n"+ prompt),"labels": row['Ground-Truth'],"task_ids":2} 


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('val.csv')

# train_df = train_df.iloc[:10,]
# test_df = test_df.iloc[:10,]

# Process tasks
# train_task1 = train_df.apply(format_data_T1, axis=1).tolist()
train_task2 = train_df.apply(format_data_T2, axis=1).tolist()
# test_task1 = test_df.apply(format_data_T1, axis=1).tolist()
test_task2 = test_df.apply(format_data_T2, axis=1).tolist()

dataset = DatasetDict({

    # 'train': Dataset.from_list(train_task1),
    # 'test': Dataset.from_list(test_task1)

    'train': Dataset.from_list(train_task2),
    'test': Dataset.from_list(test_task2)

    # 'train': Dataset.from_list(train_task1 + train_task2),
    # 'test': Dataset.from_list(test_task1 + test_task2)

})

dataset


#load the model
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
import torch

# model_name = 'unsloth/Qwen2-7B-bnb-4bit'
model_name = 'unsloth/llama-3-8b-bnb-4bit'
# model_name = "unsloth/mistral-7b-bnb-4bit"

# device=torch.device('cuda:0')   in case you want to load your model into a single device.
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    device_map='auto'
)


#loading the peft for lora
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    modules_to_save=['score'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

#tokenizing the data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1


def data_preprocesing(row):

    data = row['text']

    # Combine old and new content into a single list of sentences
    sentences = [text for text in data]
    return tokenizer(sentences, truncation=True, max_length=1024)

tokenized_data = dataset.map(data_preprocesing, batched=True)
# remove_columns=['old_content','new_content','old_time','new_time','wiki-ref'])
tokenized_data.set_format("torch")


from transformers import DataCollatorWithPadding

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


train_df = pd.read_csv('train.csv')
train_df.rename(columns={'Ground-Truth': 'labels'}, inplace=True)


#define evaluation function
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)

    metrics = {
        'Accuracy': accuracy_score(labels, predictions),
        'Precision': precision_score(labels, predictions,average='binary'),
        'Recall': recall_score(labels, predictions,average='binary'),
        'F1 Score': f1_score(labels, predictions,average='binary'),
        # 'Classification Report': classification_report(labels, predictions, output_dict=True)  # output_dict=True ensures a dict is returned
    }

    return metrics

#loading custom class for head


import torch

class_weights=(1/train_df.labels.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
class_weights


from transformers import Trainer
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    

from transformers import TrainingArguments

REPOSITORY_ID="rishavranaut/llama3_task2"

training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir = REPOSITORY_ID,
    learning_rate = 1e-4,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 8,
    weight_decay = 0.01,
    logging_strategy="steps",
    logging_steps=200,
    report_to="tensorboard",
    evaluation_strategy = 'steps',
    save_strategy='steps',
    save_steps=200, 
    load_best_model_at_end = True,
    save_total_limit=2,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id= REPOSITORY_ID,
    hub_token='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',
    resume_from_checkpoint=True
)


trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data['train'],
    eval_dataset = tokenized_data['test'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights= class_weights
)

train_result = trainer.train()
tokenizer.save_pretrained(REPOSITORY_ID)
trainer.create_model_card()
trainer.push_to_hub(REPOSITORY_ID)
gc.collect()  # Run garbage collection to free CPU RAM
torch.cuda.empty_cache()  # Clear GPU memory
print("===================INFERENCING=====================")


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from datasets import DatasetDict, Dataset
from peft import PeftModel, PeftConfig, get_peft_model



# model_name = "meta-llama/Meta-Llama-3-8B"
model_name=model_name

# Load the base model with quantization
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    device_map='auto'  # Adjust device_map as needed
)
path=REPOSITORY_ID
config = PeftConfig.from_pretrained(path)

model = PeftModel.from_pretrained(base_model,path)


import pandas as pd
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

# Setting the data
df = pd.read_csv('test.csv')

def format_data_T1(row, flag=-1):
    prompt = f"Old sentence: {row['old_cleaned_content']}\n[SEP]\nNew sentence: {row['new_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """'Determine whether a text passage has been updated by identifying changes in date, numbers, scores, statuses, or other relevant information between two given sentences. Provide a binary answer (Yes/No) indicating if the new sentence represents an update to the old sentence.'"""

    return str(base_prompt +"\n"+ prompt)

# Function to format data for Task 2
def format_data_T2(row, flag=-1):
    prompt = f"Text: {row['old_cleaned_content']}\nAnswer:"
    completion = "Yes" if row['Ground-Truth'] == 1 else "No"

    base_prompt = """Analyze the given text based on its contextual understanding to determine whether any factual updates (e.g., date changes, numerical updates, score modifications, or status changes) are likely to occur in the future. 
    Return a response indicating "Yes" if an update is predicted and "No" otherwise.\n"""

    return str(base_prompt +"\n"+ prompt)


df['text'] = df.apply(format_data_T1,axis=1)


print("============Generating Predictions================================================================")

import torch
def generate_predictions(model,df):
    old_contents = df['text']

    # Combine old and new content into a single list of sentences
    sentences = [text for text in old_contents]
    # sentences = df_test.text.tolist()
    batch_size = 2
    all_outputs = []
    
    for i in range(0, len(sentences), batch_size):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=4000)
       
        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
        for k, v in inputs.items()}
        # inputs["input_ids"][inputs["input_ids"] >= tokenizer.vocab_size] = 0 

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    df['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model,df)


print("============EVALUATING================================================================")
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report,precision_score,recall_score,confusion_matrix
labels=df['Ground-Truth'].astype(int)
predictions=df['predictions'].astype(int)

print('accuracy', accuracy_score(labels, predictions))
print('precision', precision_score(labels, predictions, average='binary'))
print('recall', recall_score(labels, predictions, average='binary'))
print('f1_score', f1_score(labels, predictions, average='binary'))
print('classification_report', classification_report(labels, predictions))

results = pd.DataFrame({'actual':labels,'prediciton':predictions})

save_in = REPOSITORY_ID.split('/')[1]
import pickle
with open(f'{save_in}.pkl', 'wb') as f:
    pickle.dump(results, f)