# NarrativeKoGPT2 ref : Fine tuning
from old.data_newphrase import PhraseDataSet
import os
import random
import torch
import math
import argparse
from tqdm import tqdm

from old.data_tokenizermrbnn import MyTokenizer
from old.util import *
from torch.utils.data import DataLoader
from transformers import (GPT2Config, 
                          GPT2LMHeadModel, 
                          PreTrainedTokenizerFast,
                          DataCollatorForLanguageModeling,
                          LineByLineTextDataset,
                          TrainingArguments, 
                          Trainer)
from data_novel import NovelDataSet


def main_generaltrain(config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    if config['input_path'] is not None:
        model = GPT2LMHeadModel(config=GPT2Config(vocab_size=52000))
        model.load_state_dict(torch.load(config['input_path']), strict=False)
    else:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    # model.config ?
    model.to(device) 
    
    if config['tokenizer'] == 'kogpt2':
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
    elif config['tokenizer'] == 'mrbnn': # MrBananaman tokenizer
        vocab_file_path = './old/tokenizer/vocab.json'
        merge_file_path = './old/tokenizer/merges.txt'
        tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
        # SPECIAL_TOKEN = ['<s>', '</s>']
        # add_special_tokens_(model, tokenizer, SPECIAL_TOKEN)
    else:
        raise ValueError('Not defined tokenizer.')
    
    learning_rate = config['lr']
    epochs = config['epoch']
    batch_size = config['batch_size']  # 4
    
    file_path = 'preprocess_final.csv'
    dataset = NovelDataSet(file_path, tokenizer)
    
    novel_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=True)
    count = 0
    avg_loss = (0.0, 0.0)
    
    loss_dir = config['loss_dir']
    if not os.path.isdir(loss_dir):
        os.mkdir(loss_dir)
    loss_file = os.path.join(loss_dir, 'loss_new.txt')
    
    weight_dir = config['weight_dir']
    if not os.path.isdir(weight_dir):
        os.mkdir(weight_dir)
    

    prefix = config['prefix_weight']
    output_freq = config['output_freq']
    total_count = 0
    for epoch in tqdm(range(epochs)):
        count = 0
        for batch in novel_dataloader:
            # print('start training..')
            optimizer.zero_grad()
            
            # batch = torch.stack(batch)
            batch = batch.transpose(1, 0)  # batch (1024, 2)
            batch = batch.to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs[0]
            # loss, logits = outputs[:2]  # ??
            # loss.to(device)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            loss = loss.item()
            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
            
            if (count + 1) % output_freq == 0:
                print('epoch {0} train_iteration {1} | loss = {2:.5f} avg_loss = {3:.5f}'.format(epoch, count, loss,
                                                                                                 avg_loss[0] / avg_loss[1]))
                l = 'epoch' + str(epoch) + 'train_iteration' + str(count) + ' | loss: ' + str(
                    loss) + 'avg_loss: ' + str(avg_loss)
                with open(loss_file, 'a') as f:
                    f.write(l)
                    f.write('\n')
            count += 1
            total_count += 1
        
        torch.save({'epoch': epoch, 'model state_dict': model.state_dict()}, weight_dir + prefix + str(epoch) + '.bin')
    
def main(config):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained('weight/trainer_msl256_ep3+47/checkpoint-191000')
    # model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # dataloader for batch
    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='preprocess_final.txt', block_size=256)

    train_args = TrainingArguments(
        output_dir = 'weight/trainer_msl256_ep50+50',
        overwrite_output_dir = True,
        dataloader_drop_last = True,
        per_device_train_batch_size = config['batch_size'],
        learning_rate = config['lr'],
        num_train_epochs = config['epoch'],
        save_total_limit=2
    )
    trainer = Trainer(
        model = model,
        args = train_args,
        data_collator = data_collator,
        train_dataset = dataset
    )
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', default=47, type=int,
                        dest='epoch', help='training epoch')
    parser.add_argument('--learning-rate', '-lr', default=5e-5, type=float,
                        dest='lr', help='training learning rate')
    parser.add_argument('--batch-size', '-bs', default=4, type=int,
                        dest='batch_size', help='training batch size')
    parser.add_argument('--loss-dir', '-ld', default='loss_log/', type=str,
                        dest='loss_dir', help='Path to save log for training loss')
    parser.add_argument('--weight-dir', '-wd', default='weight/', type=str,
                        dest='weight_dir', help='Path to save weight of model')
    parser.add_argument('--weight-file', '-wf', default='fine_novel_', type=str,
                        dest='prefix_weight', help='Prefix for weight files')
    parser.add_argument('--output-frequency', '-of', default=40, type=int,
                        dest='output_freq', help='Frequency of results of loss')
    parser.add_argument('--tokenizer', '-t', default='kogpt2', type=str,
                        dest='tokenizer', help='Type of tokenizer(kogpt2, mrbnn)')
    parser.add_argument('--input-weight', '-i', default=None, type=str,
                        dest='input_path', help='Pre-trained weight')
    args = parser.parse_args()
    
    config = {
        'epoch': args.epoch,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'loss_dir': args.loss_dir,
        'weight_dir': args.weight_dir,
        'prefix_weight': args.prefix_weight,
        'output_freq': args.output_freq,
        'tokenizer': args.tokenizer,
        'input_path': args.input_path
    }
    
    main(config)