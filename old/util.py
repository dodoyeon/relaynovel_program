# from transformers import GPT2LMHeadModel, GPT2Config
# from transformers import PreTrainedTokenizerFast
# from data_tokenizermrbnn import MyTokenizer
# from os import listdir, mkdir
# from os.path import isfile, isdir, join
import torch

# Generation code
def encoding(text, tokenizer):
    tokens = ['<s>'] + tokenizer.tokenize(text)# + ['</s>']
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids, tokenizer):
    return tokenizer.decode(ids[0])
    # return tokenizer.convert_ids_to_tokens(ids[0])

# Tokenizer
def add_special_tokens_(model, tokenizer, SPECIAL_TOKEN):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(SPECIAL_TOKEN)
    num_add_tokens = len(SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_add_tokens + 1)