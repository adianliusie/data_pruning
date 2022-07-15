#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraForMultipleChoice
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

from rankers.ranker import make_ranker

MAXLEN = 256

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--train_data_path', type=str, help='Load path of training data')
parser.add_argument('--val_data_path', type=str, help='Load path of validation data')
parser.add_argument('--val_interval', type=int, default=1, help='Epoch intervals for evaluating on validation set')
parser.add_argument('--save_path', type=str, help='Load path to which best trained model will be saved')
parser.add_argument('--ranking_type', type=str, default='random', help='Ranking order for training data')
parser.add_argument('--data_frac', type=float, default=0.1, help='Specify the fraction of data to use')
parser.add_argument('--ext_model_path', type=str, default=None, help='trained model to use for pruning')


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def rank(labels, input_ids, token_type_ids, attention_masks, ranking_type = 'random', model_path=None, ret_frac=1.0, device=None):

    # if ranking_type == 'random':
    #     combo = list(zip(labels, input_ids, token_type_ids, attention_masks))
    #     random.shuffle(combo)
    #     labels, input_ids, token_type_ids, attention_masks = zip(*combo)
    # else:
    data = [{'inputs':[i, t, a], 'output':l} for i,t,a,l in zip(input_ids, token_type_ids, attention_masks, labels)]
    data = data[:8]
    ranker = make_ranker(ranking_type, model_path=model_path, device=device)
    out_data = ranker.filter_data(data, ret_frac=ret_frac)

    input_ids = [d['inputs'][0] for d in out_data]
    token_type_ids = [d['inputs'][1] for d in out_data]
    attention_masks = [d['inputs'][2] for d in out_data]
    labels = [d['output'] for d in out_data]
    return labels, input_ids, token_type_ids, attention_masks


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    with open(args.train_data_path) as f:
        train_data = json.load(f)

    labels = []
    input_ids = []
    token_type_ids = []

    for item in train_data:
        context = item["context"]
        question = item["question"]
        lab = item["label"]
        labels.append(lab)
        four_inp_ids = []
        four_tok_type_ids = []
        for i, ans in enumerate(item["answers"]):
            combo = context + " [SEP] " + question + " " + ans
            inp_ids = tokenizer.encode(combo)
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            four_inp_ids.append(inp_ids)
            four_tok_type_ids.append(tok_type_ids)
        four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        input_ids.append(four_inp_ids)
        token_type_ids.append(four_tok_type_ids)
    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        sen_attention_masks = []
        for opt in sen:
            att_mask = [int(token_id > 0) for token_id in opt]
            sen_attention_masks.append(att_mask)
        attention_masks.append(sen_attention_masks)

    with open(args.val_data_path) as f:
        val_data = json.load(f)

    labels_val = []
    input_ids_val = []
    token_type_ids_val = []

    for item in val_data:
        context = item["context"]
        question = item["question"]
        lab = item["label"]
        labels_val.append(lab)
        four_inp_ids = []
        four_tok_type_ids = []
        for i, ans in enumerate(item["answers"]):
            combo = context + " [SEP] " + question + " " + ans
            inp_ids = tokenizer.encode(combo)
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            four_inp_ids.append(inp_ids)
            four_tok_type_ids.append(tok_type_ids)
        four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        input_ids_val.append(four_inp_ids)
        token_type_ids_val.append(four_tok_type_ids)
    # Create attention masks
    attention_masks_val = []
    for sen in input_ids_val:
        sen_attention_masks = []
        for opt in sen:
            att_mask = [int(token_id > 0) for token_id in opt]
            sen_attention_masks.append(att_mask)
        attention_masks_val.append(sen_attention_masks)

    # Apply some ranking operation
    labels, input_ids, token_type_ids, attention_masks = rank(labels, input_ids, token_type_ids, attention_masks, ranking_type = args.ranking_type, model_path=args.ext_model_path, ret_frac=args.data_frac, device=device)


    # # Keep the best examples
    # num_examples = len(labels)
    # to_keep = int(num_examples * args.data_frac)
    # labels = labels[:to_keep]
    # input_ids = input_ids[:to_keep]
    # token_type_ids = token_type_ids[:to_keep]
    # attention_masks = attention_masks[:to_keep]

    # Convert to torch tensors
    labels = torch.tensor(labels)
    labels = labels.long().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    labels_val = torch.tensor(labels_val)
    labels_val = labels_val.long().to(device)
    input_ids_val = torch.tensor(input_ids_val)
    input_ids_val = input_ids_val.long().to(device)
    token_type_ids_val = torch.tensor(token_type_ids_val)
    token_type_ids_val = token_type_ids_val.long().to(device)
    attention_masks_val = torch.tensor(attention_masks_val)
    attention_masks_val = attention_masks_val.long().to(device)

    train_data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    val_data = TensorDataset(input_ids_val, token_type_ids_val, attention_masks_val, labels_val)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = ElectraForMultipleChoice.from_pretrained(electra_large).to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    loss_values = []

    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps)


    best_val_loss = 100
    file_path = args.save_path+'electra_QA_MC_seed'+str(args.seed)+'.pt'

    for epoch in range(args.n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_tok_typ_ids = batch[1].to(device)
            b_att_msks = batch[2].to(device)
            b_labs = batch[3].to(device)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids, labels=b_labs)
            loss = outputs[0]
            total_loss += loss.item()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            total_loss_val = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    b_input_ids = batch[0].to(device)
                    b_tok_typ_ids = batch[1].to(device)
                    b_att_msks = batch[2].to(device)
                    b_labs = batch[3].to(device)
                    outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids, labels=b_labs)
                    loss = outputs[0]
                    total_loss_val += loss.item()
                avg_val_loss = total_loss_val / len(val_dataloader)
            print("  Average validation loss: {0:.2f}".format(avg_val_loss))
            if avg_val_loss < best_val_loss:
                torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)