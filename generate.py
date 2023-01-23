from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

def test_pre(sentence):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    input_ids = tokenizer.encode(sentence)
    gen_ids = model.generate(torch.tensor([input_ids]),
                            max_length=128,
                            repetition_penalty=2.0,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=True)
    generated = tokenizer.decode(gen_ids[0,:].tolist())
    return generated

def generate(sentence, model_dir):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                        bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')
    pre_model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model = GPT2LMHeadModel(pre_model.config)
    model.load_state_dict(torch.load(model_dir)['model state_dict']) # , strict=False
    model.eval()
    input_ids = tokenizer.encode(sentence)
    gen_ids = model.generate(torch.tensor([input_ids]),
                            # do_sample=True,
                            max_length=128,
                            no_repeat_ngram_size=2,
                            # top_k=50,
                            # top_p=0.95,
                            repetition_penalty=1.2,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=True)
    generated = tokenizer.decode(gen_ids[0,:].tolist())
    return generated

input_sent = '외계인 경수는 오늘 아침에 일어나 판타지 소설을 쓰고 있습니다.'
model_dir = 'weight/msl256_ep2.bin'
# print(test_pre(input_sent))
print(generate(input_sent, model_dir))