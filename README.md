# English to Telugu Translation Project

## Overview
This project focuses on translation from English to Telugu using the machine translation models. We have fine-tuned the following models on the Samanantar dataset:
- MBART50
- MT5
- NLLB
- Llama-7B

We selected a subset of 60,000 parallel corpora from the available 5 million parallel corpora for our training data  and FLORES22 as evaluation set.

## Models
### [MBART50](https://arxiv.org/pdf/2008.00401.pdf)

#### Model Overview:
[mBART50](https://arxiv.org/pdf/2008.00401.pdf) (Multilingual BART) is a multilingual extension of the BART (Bidirectional and Auto-Regressive Transformers) model. It is specifically designed for multilingual sequence-to-sequence tasks and demonstrates strong performance across various languages. By pretraining on multiple languages simultaneously, mBART learns a shared cross-lingual representation space, enabling effective transfer learning across languages.

#### Fine-Tuning Process:
For [fine-tuning mBART](https://huggingface.co/transformers/v4.7.0/model_doc/mbart.html) on our task, we utilized the Samanantar dataset. This dataset is rich in multilingual text data, making it suitable for training multilingual models like mBART. The fine-tuning process involves adapting the pretrained mBART model to our specific task by updating its parameters based on the Samanantar dataset. We have fine-tuned the model for only one epoch and were able to achieve a SacreBLEU score of approximately 9 using a batch size of 8, a learning rate of 5e-4, and the Adam optimizer. We set MAX_INPUT_LENGTH and MAX_TARGET_LENGTH to 128. The training stage took around 35 minutes for one epoch on one NVIDIA A100 GPU. We observed that training for more epochs could increase the quality of translation. The code is present here [link](./MBart50model.py)


### [MT5](https://arxiv.org/abs/2010.11934)

#### Model Overview:
MT5 (Multilingual T5) is a multilingual variant of the T5 (Text-To-Text Transfer Transformer) model, which is a powerful text-to-text transformer architecture. MT5 is designed to handle multilingual tasks, allowing for seamless processing of text in multiple languages within a single model. By pretraining on a diverse set of languages, MT5 learns to understand and generate text in various languages, enabling effective transfer learning across linguistic boundaries.

#### Fine-Tuning Process:
To [fine-tune MT5](https://huggingface.co/docs/transformers/v4.14.1/en/model_doc/mt5) for our specific task, we followed a similar approach to the pretrained mBART model. We utilized a multilingual dataset suitable for our task and fine-tuned MT5 on this dataset. The fine-tuning process involves updating the parameters of the pretrained MT5 model based on the task-specific dataset, enabling the model to adapt its knowledge to our target task.  The code is present here [link](./mT5model.py)


### [NLLB](https://arxiv.org/abs/2207.04672)
- For details about NLLB see - https://huggingface.co/docs/transformers/en/model_doc/nllb
- We [fine-tuned](https://discuss.huggingface.co/t/fine-tuning-nllb-model/31237) NLLB model after preprocessing unknown tokens in Telugu language on Samanantar dataset with configurations as follows:
1)Train split= 0.8 , test split = 0.2 on subset of Samanatar dataset 
2)used Adam with initial lr= 2e-4 and batchsize of 8.
3)epoch =1.

  The code is present here [link](./NLLB_finetunedmodel.ipynb)

### [Llama-7B](https://arxiv.org/abs/2307.09288)
- Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 7B pretrained model.
-  We hypothesized that a bottleneck in performance could occur for low-resource languages like Telugu due to the tokenizer. The Llama tokenizer had a vocabulary size of 32k tokens, with fewer tokens dedicated to low-resource languages like Telugu. we observed that each Telugu word was divided into an average of 20.03 tokens in the Samanantar dataset which is worse, whereas only 1.6 for English. To address this, we added 20k new quality tokens to the Llama tokenizer, sourced from the Telugu version of CulturaX.
- we used the Sentence Piece tokenizer on the CulturaX dataset for generating new quality tokens. these tokens were then integrated into the Llama tokenizer, and the Llama pretrained model embedding layer and softmax layers were resized accordingly.

#### Fine-Tuning Process:
- For fine-tuning we have considered the alpaca dataset, which typically consists of the following format Instruction,input,response.
- PreTraining and fine-tuning phase is happened in a lightweight finetuning paradigm called LoRA.

The Llama model is initially loaded in an 8-bit quantization format, which offers efficiency in terms of memory usage. Moreover, it has been demonstrated that this quantization method does not significantly compromise the model’s performance.

For more details about Llama-7B - https://huggingface.co/meta-llama/Llama-2-7b
 

# Evaluation Metrics for Machine Translation

In this project, we consider various evaluation metrics to assess the performance of machine translation systems. These metrics provide insights into different aspects of translation quality, including fluency, adequacy, word-level accuracy, and sentence-level characteristics. The following metrics are considered:

## 1. Aggregate Scores
- **BLEU Score**: Measures the similarity between the candidate translation and reference translations using n-gram overlap.

## 2. Word Accuracy Analysis
- **F-measure by Frequency Bucket**: Evaluates the accuracy of word predictions based on the frequency of words in the reference translation.

## 3. Sentence Bucket Analysis
- **Bucketing Sentences**: Groups sentences based on various statistics such as sentence BLEU, length difference with the reference, and overall length.
- **Statistics by Bucket**: Calculates statistics (e.g., number of sentences, BLEU score) for each bucket to analyze translation quality across different sentence characteristics.

## 4. N-gram Difference Analysis
- **Consistency of N-gram Translation**: Identifies which n-grams one system consistently translates better than the other.

## 5. Sentence Examples
- **Identifying Superior Translations**: Finds sentences where one system performs better than the other according to sentence BLEU.

These evaluation metrics provide a comprehensive analysis of machine translation systems, enabling us to identify strengths and weaknesses and guide improvements in translation quality. They are valuable tools for researchers and developers working on machine translation models.


## Data
A data sample of 500 parallel corpus of English and Telugu sentences  taken from [Samanantar](https://ai4bharat.iitm.ac.in/samanantar/) dataset  is available in the [`data`](./Data).


## Conclusion and Future Work
In this project, we have exploited LLMs like Llama2, mBART50, NLLB and mt5 and tried fine-tuning the models effectively using a small training size of 60000 samples taken from the Samanantar dataset and were able to obtain a decent increase in BLEU score. We have observed repeating the pretraining step can help LLMs to perform better than direct finetuning while using custom tokenizers. We have observed the importance of data that is been used for fine-tuning setup to build better models. We have found that LLMs have the potential
to perform translations involving English and
Telugu , which distinguishes
them from traditional translation models. Particularly, LLaMA-2 based models exhibit superior performance in zero-shot and in-context example-based learning, highlighting their effectiveness in cross-lingual tasks.

For future work, one can anticipate achieving greater performance enhancements by obtaining higher-quality training data and conducting meticulous training sessions with careful hyperparameter tuning.One can aim to boost performance by training on the complete Samanantar dataset (50,00,000 examples) and exploring one-shot or few-shot tuning of Llama3-8b, a chat-based model, to enhance Telugu translations. Additionally, expanding the Telugu tokenizer to include more tokens can augment the number of available tokens, potentially enhancing the model's ability to capture finer linguistic nuances and improve translation accuracy.

One can aim to explore pre-training on Indic language monolingual corpora from IndicCorp and develop a single-script mT5- or mBART-like model for Indic languages. Subsequently, fine-tuning on machine translation tasks using the Samanantar dataset can lead to more effective translation models tailored to Indic languages.And also one can explore various data filtering techniques and data sampling methods  to assess their impact on the performance of the Machine Translation (MT) model during training. For instance, one can aim to experiment with sorting data sources by quality and selectively feeding sentences from high-quality sources in later epochs of training to enhance model performance.


## Acknowledgements
This project is contributed to by the following collaborators:

- [Kodamanchili Tarun](https://github.com/kodamanchili-tarun)


## References 

Note: We have made an effort to refer to as many sources as possible for this document. We apologize for any oversights.



