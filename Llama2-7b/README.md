# Fine-Tuning Llama-7b Model for Telugu Translation

Welcome to the repository for fine-tuning the Llama-7b model for Telugu translation! This README will guide you through the file structure and steps to get started with fine-tuning and using the model for Telugu translation.

## File Structure

The repository is organized into the following directories:

1. **FineTuning**: Contains scripts and files related to fine-tuning the Llama-7b model for Telugu translation.

   - `translate_alpaca_dataset_to_telugu.py`: Translates the Alpaca dataset used for fine-tuning into Telugu.
   - `finetune.py`: Code for supervised fine-tuning after loading the Pretrained model either from local storage or Hugging Face.

2. **Pretraining**: Includes any pretraining scripts or resources used for the Llama-7b model.

   - `download_base_model`: Downloads the base model from Hugging Face for pretraining.
   - `script.sh`: Pretraining parameter arguments script.
   - `run_pretraining.py`: Code for PreTraining the downloaded model on the datasets.
   - `merge_lora_with_llama.py`: Merges the learned weights from lightweight finetuning (LORA) with the original model.

3. **Tokenizer**: Contains scripts or files related to tokenization specific to Telugu language.

   - `tokenizer.py`: Scrapes tokens from the specified dataset using the Sentence Piece Trainer.
   - `merge_tokenizer.py`: Merges the tokens scraped above with the original downloaded model.

4. **requirements.txt**: Lists the dependencies and libraries required to run the code successfully.

## Getting Started

To get started with fine-tuning the Llama-7b model for Telugu translation, follow these steps:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/llama-7b-telugu-translation.git
   ```

2. **Install Dependencies**: Navigate to the cloned directory and install the required dependencies using `pip`.

   ```bash
   cd llama-7b-telugu-translation
   pip install -r requirements.txt
   ```
3. **Prepare Data**: Create a folder named 'data' and add the datasets you want to pretrain and fine-tune on.

4. **Tokenization (Optional)**: If you need custom tokenization for Telugu language:
   - Run `python tokenizer.py`
   - Run `python merge_tokenizer.py`

5. **Pretraining (Optional)**: If you want to perform pretraining on the Llama-7b model:
   - Run `python download_base_llama2-7b-hf.py`
   - Run `./script.sh` (This will execute `run_pretrain.py` and perform pretraining)
   - Run `python merge_lora_with_llama.py`
   
6. **Fine-Tune the Model**: Use the scripts provided in the `FineTuning` directory to fine-tune the Llama-7b model for Telugu translation:
   - Run `python translate_alpaca_dataset_to_telugu.py`
   - Run `python finetune.py`

7. **Usage**: Once fine-tuned, you can use the model for Telugu translation tasks. Ensure you publish it to Hugging Face or follow specific deployment instructions.

## Contributing

If you'd like to contribute to this repository, feel free to fork it, make your changes, and submit a pull request. We welcome contributions and improvements to make the fine-tuning process smoother for Telugu translation tasks.

## Issues

If you encounter any issues or have suggestions for improvements, please open an issue on GitHub. We appreciate your feedback and will work to address any problems promptly.

Happy fine-tuning and translating in Telugu with the Llama-7b model! 🌟
