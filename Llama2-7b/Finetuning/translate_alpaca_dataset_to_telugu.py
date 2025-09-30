import pandas as pd
import googletrans
from googletrans import Translator
from datasets import load_dataset
import concurrent.futures
import subprocess

import time

def translate_text(index, text):
    print(" {} of {} completed".format(index, len(text)))
    translator = Translator()
    time.sleep(1)
    try:
        translated_text = translator.translate(text, src='en', dest='te')
        if translated_text is None:
            print(f"No translation returned for text at index {index}")
            return None
        return translated_text.text
    except Exception as e:
        print(f"Error translating text at index {index}: {e}")
        return None

def main():
    dataset = load_dataset("tatsu-lab/alpaca")
    mlm_lst = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(translate_text, i, text) for i, text in enumerate(dataset["train"]["text"])]

        for future in concurrent.futures.as_completed(futures):
            if future.result() is not None:
               mlm_lst.append(future.result())

    df = pd.DataFrame(mlm_lst, columns=['Prompt'])
    print("completed")
    df.to_csv('translated_eng2tel.csv', encoding="utf-8")

if __name__ == "__main__":
    main()
