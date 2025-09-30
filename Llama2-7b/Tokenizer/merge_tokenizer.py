import os

### os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

from huggingface_hub import login

### load
telugu_sp_model_file = "TeluguLLM.model"
llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

telugu_sp_model = spm.SentencePieceProcessor()
telugu_sp_model.Load(telugu_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
telugu_spm = sp_pb2_model.ModelProto()
telugu_spm.ParseFromString(telugu_sp_model.serialized_model_proto())

### print number of tokens
print(len(llama_tokenizer), len(telugu_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

### Add new Telugu tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")

for p in telugu_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

### Save
output_sp_dir = "merged_tokenizer_sentencepiece"
output_hf_dir = "merged_tokenizer_huggingface" 
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/telugu_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())

tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/telugu_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"Telugu-LLaMA tokenizer has been saved to {output_hf_dir}")

### Test
telugu_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = """ హాయ్ ఎలా ఉన్నారు నిన్న రాత్రి ఎక్కడికి వెళ్తున్నావు Can India secure the World Cup trophy?"""
print("Test text:\n", text)
llama_tokenized = llama_tokenizer.tokenize(text)
telugu_llama_tokenized = telugu_llama_tokenizer.tokenize(text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenized}")
print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
print(f"Tokenized by telugu-LLaMA tokenizer:{telugu_llama_tokenized}")
print(f"telugu LLaMA tokenizer n_tokens={len(telugu_llama_tokenized)}")