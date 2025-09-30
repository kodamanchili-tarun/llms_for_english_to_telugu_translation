import sentencepiece as spm

## Training the tokenizer using SentencePiece , bpe(byte pair encoding)
spm.SentencePieceTrainer.train(
            input="/raid/home/prabhasreddy/nlp_fine/fine_tune_lama/TeluguLLM/data/te/te.txt", #data path
            model_prefix="TeluguLLM",                                                         #model name
            vocab_size=20000,       #number of entra tokens to be added
            character_coverage=1.0,
            model_type="bpe",
        )


# To load and test the tokenizer
sp = spm.SentencePieceProcessor()
sp.load('TeluguLLM.model')
print(sp.encode_as_pieces('నా పేరు ప్రభాస్ రెడ్డి , నేను హైదరాబాద్ లో ఉన్నాను'))


