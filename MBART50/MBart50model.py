# %% [markdown]
# # **Fine Tuning Transformer for Neural Machine Translation**

# %% [markdown]
# ## Installing Dependencies

# %%
# # ! pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package_name>
# %pip install datasets sacrebleu torch transformers sentencepiece transformers[sentencepiece]
# %pip cache purge
# %pip install accelerate -U
# ! pip install accelerate -U
# %pip install peft
# %pip install trl
# %pip install datasets


# %% [markdown]
# ## Required Imports

# %%
import warnings
import numpy as np
import pandas as pd

import torch
import transformers

from datasets import Dataset
from datasets import load_metric
from transformers import Trainer
import accelerate


from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM,AutoModel
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizer,
     Seq2SeqTrainingArguments, Seq2SeqTrainer
   )

warnings.filterwarnings("ignore")
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.set_device(2)

# %% [markdown]
# ## Constants

# %%
file_path_english = './eng.txt'
file_path_telugu = './tel.txt'
english_sentances = []
telugu_sentances=[]


with open(file_path_english, 'r', encoding='utf-8') as file:
    for line in file:
        
        english_sentances.append(line.strip())

with open(file_path_telugu, 'r', encoding='utf-8') as file:
    for line in file:
        
        telugu_sentances.append(line.strip())

translation_data = pd.DataFrame({"english_text":english_sentances,"telugu_text":telugu_sentances})


# %% [markdown]
# ## We  will evaluate our project on FLORES22 Dataset also

# %%
file_path_english = './FLORES22deveng.txt'
file_path_telugu = './FLORES22devtel.txt'
english_sentances = []
telugu_sentances=[]


with open(file_path_english, 'r', encoding='utf-8') as file:
    for line in file:
        
        english_sentances.append(line.strip())

with open(file_path_telugu, 'r', encoding='utf-8') as file:
    for line in file:
        
        telugu_sentances.append(line.strip())

evaluation_data = pd.DataFrame({"english_text":english_sentances,"telugu_text":telugu_sentances})


# %%
BATCH_SIZE = 2
BLEU = "bleu"
ENGLISH = "en"
ENGLISH_TEXT = "english_text"
EPOCH = "epoch"
INPUT_IDS = "input_ids"
GEN_LEN = "gen_len"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
MODEL_CHECKPOINT = "facebook/mbart-large-cc25"
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
LABELS = "labels"
PREFIX = ""
TELUGU = "te"
TELUGU_TEXT = "telugu_text"
SCORE = "score"
SOURCE_LANG = "en"
TARGET_LANG = "te"
TRANSLATION = "translation"
UNNAMED_COL = "Unnamed: 0"

# %% [markdown]
# ## Helper Functions

# %%
def postprocess_text(preds: list, labels: list) -> tuple:
    """Performs post processing on the prediction text and labels"""

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[TRANSLATION] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[ENGLISH] = sr_text
        temp_dict[TELUGU] = tr_text

        data_dict[TRANSLATION].append(temp_dict)

    return data_dict


def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 
                                 tokenizer: MBart50TokenizerFast) -> list:
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs =  row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
        
        model_inputs[TRANSLATION] = row

    
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
            
            model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)
        
        # if targets is not None:
        #     labels = label_tokenizer(targets, max_length=MAX_INPUT_LENGTH,
        #                        truncation=True, padding=True)

        #     model_inputs[LABELS] = labels['input_ids']
        # else:
        #     # Handle missing targets gracefully
        #     model_inputs[LABELS] = None

        # preped_data.append(model_inputs)

    return preped_data


tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX",tgt_lang="te_IN")
    
def compute_metrics(eval_preds: tuple) -> dict:
    """computes bleu score and other performance metrics """

    metric = load_metric("sacrebleu")
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25",src_lang="en_XX")
    # tokenizer=MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25",src_lang="te_IN")
    
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX",tgt_lang="te_IN")
    
    # tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {BLEU: result[SCORE]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result[GEN_LEN] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result

# %% [markdown]
# ## Loading and Preparing The Dataset

# %%
# As of now, we are using only 60,000 records for training
translation_data = translation_data[:60000]


# %% [markdown]
# ## Train, Test & Validation Split of Data

# %%
#creating the evaluation dataset from the evaluation data which comes from FLORES22 dataset
x_eval=evaluation_data["english_text"]
y_eval=evaluation_data["telugu_text"]
eval_data = prep_data_for_model_fine_tuning(x_eval, y_eval)
eval_data

# %%
X = translation_data[ENGLISH_TEXT]
y = translation_data[TELUGU_TEXT]

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                    shuffle=True,
                                                    random_state=100)

print("INITIAL X-TRAIN SHAPE: ", x_train.shape)
print("INITIAL Y-TRAIN SHAPE: ", y_train.shape)
print("X-TEST SHAPE: ", x_test.shape)
print("Y-TEST SHAPE: ", y_test.shape)

# %%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.20,
                                                  shuffle=True,
                                                  random_state=100)

print("FINAL X-TRAIN SHAPE: ", x_train.shape)
print("FINAL Y-TRAIN SHAPE: ", y_train.shape)
print("X-VAL SHAPE: ", x_val.shape)
print("Y-VAL SHAPE: ", y_val.shape)

# %% [markdown]
# ## Load Tokenizer from AutoTokenizer Class

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline,MT5ForConditionalGeneration, T5Tokenizer


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX",tgt_lang="te_IN")

# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

# base_model_name = "NousResearch/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
    
# )
# tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


# %% [markdown]
# ## Prepare the model ready dataset

# %%
training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)

validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)


test_data = prep_data_for_model_fine_tuning(x_test.values, y_test.values)

# %%
training_data[TRANSLATION]

# %%
train_data = generate_model_ready_dataset(dataset=training_data[TRANSLATION],
                                          tokenizer=tokenizer,
                                          
                                          source=ENGLISH,
                                          target=TELUGU)


validation_data = generate_model_ready_dataset(dataset=validation_data[TRANSLATION],
                                               tokenizer=tokenizer,
                                               
                                               source=ENGLISH,
                                               target=TELUGU
                                               )

test_data = generate_model_ready_dataset(dataset=test_data[TRANSLATION],
                                               tokenizer=tokenizer,
                                               
                                               source=ENGLISH,
                                               target=TELUGU
                                              )

eval_data = generate_model_ready_dataset(dataset=eval_data[TRANSLATION],
                                         tokenizer=tokenizer,
                                         source=ENGLISH,
                                         target=TELUGU)

# %%
train_data[0]

# %%
train_df = pd.DataFrame.from_records(train_data)
train_df.info()

# %%
validation_df = pd.DataFrame.from_records(validation_data)
validation_df.info()

# %%
test_df = pd.DataFrame.from_records(test_data)
test_df.info()

# %%
eval_df = pd.DataFrame.from_records(eval_data)
eval_df.info()

# %% [markdown]
# ## Convert dataframe to Dataset Class object

# %%
train_df

# %%
train_dataset = Dataset.from_pandas(train_df)
train_dataset

# %%
validation_dataset = Dataset.from_pandas(validation_df)
validation_dataset

# %%
test_dataset = Dataset.from_pandas(test_df)
test_dataset

# %%
eval_dataset=Dataset.from_pandas(eval_df)
eval_dataset

# %% [markdown]
# ## Load model, Create Model Training Args and Data Collator

# %%
# model = AutoModel.from_pretrained(MODEL_CHECKPOINT)

# %%
model

# %%
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming `model` is your PyTorch model
num_params = count_parameters(model)
print(f"Number of trainable parameters in the model: {num_params}")


# %%
model_args = Seq2SeqTrainingArguments(
    f"{MODEL_NAME}-finetuned-{SOURCE_LANG}-to-{TARGET_LANG}",
    evaluation_strategy=EPOCH,
    learning_rate=5e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.02,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %% [markdown]
# # Fine Tuning the Model, finally !!

# %%
model.to("cuda:0")

# %%
train_dataset

# %%


trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



# %%
import gc 
gc.collect()

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %%
import torch

# Clear GPU memory data
torch.cuda.empty_cache()


# %%
trainer.train()

# %% [markdown]
# ## Saving the Fine Tuned Transformer

# %%
trainer.save_model("FineTunedTransformer")

# %% [markdown]
# ## Perform Translation on Test Datset

# %%
torch.cuda.empty_cache()
# test_results = trainer.predict(test_dataset)
test_results = trainer.predict(eval_dataset)

# %%
# print("Test Bleu Score: ", test_results.metrics["test_bleu"])
print("Eval Bleu Score: ", test_results.metrics["test_bleu"])

# %% [markdown]
# ## Generate Prediction Sentences

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
predictions = []
test_input = test_dataset[TRANSLATION]

for input_text in tqdm(test_input):
    source_sentence = input_text[ENGLISH]
    encoded_source = tokenizer(source_sentence,
                               return_tensors='pt',
                               padding=True,
                               truncation=True)
    encoded_source.to(device)  # Move input tensor to the same device as the model

    translated = model.generate(**encoded_source,forced_bos_token_id=tokenizer.lang_code_to_id["te_IN"])

    predictions.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])

# Move the model back to CPU if needed
model.to("cpu")

# %%
y_true_en = []
y_true_te = []

for input_text in tqdm(test_input):
    y_true_en.append(input_text[ENGLISH])
    y_true_te.append(input_text[TELUGU])

# %%
print(predictions)

# %%
output_df = pd.DataFrame({"y_true_english": y_true_en, "y_true_te": y_true_te, "predicted_text": predictions})
output_df


