"""
This module defines a tokenizer for the model, which is based on the Hugging Face Tokenizers library. 
It creates a WordLevel tokenizer and trains it on the provided training data, which includes special 
tokens for the model's input and output formats. The tokenizer is then saved to a specified path and 
can be loaded for use in the model.
"""
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
import os
import json
import string

SPECIALS = [
    "<|bos|>",
    "<|unk|>",
    "<|pad|>",
    "<|eos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|image|>",     # single image placeholder
    "<|image_start|>",
    "<|image_end|>",
    "<|question_start|>",
    "<|question_end|>",
    "<|str_start|>",
    "<|str_end|>",
]

def create_tokenizer(tokenizer_path, data_path='./training_data_formal.jsonl'):
    if not os.path.exists(tokenizer_path):
        with open(data_path, 'r') as f:
            if data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        tokenizer = Tokenizer(WordLevel(unk_token="<|unk|>"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        trainer = WordLevelTrainer(special_tokens=SPECIALS)

        """
        this initilizaion covers the following tokens:
        t/f, individual number tokens in output, and all chars that appear in the input string, 
        which is enough for the model apart from the question prompting.
        """
        str_list = [] + list(string.ascii_lowercase) + list(string.digits) + ['True', 'False']
        d = data[0]['question']
        str_list.append(d["text_only_num"])
        str_list.append(d["text_only_tf"])
        str_list.append(d["multimodal_num"])
        str_list.append(d["multimodal_tf"]) 
            
        tokenizer.train_from_iterator(str_list, trainer=trainer)
        tokenizer.save(tokenizer_path)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path,
                                                bos_token="<|bos|>",
                                                eos_token="<|eos|>",
                                                unk_token="<|unk|>",
                                                pad_token="<|pad|>",
                                                # image_token="<|image|>",
                                                padding_side="right",
                                                additional_special_tokens=[t for t in SPECIALS if t not in 
                                                            ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]]
                                        )
    # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer