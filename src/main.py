from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from Transformer import MyTransformer
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    # dataset store german and english tokens
    def __init__(self, de_tokenized, en_tokenized) -> object:
        self.de_data = de_tokenized
        self.en_data = en_tokenized

    def __len__(self):
        return len(self.de_data)

    def __getitem__(self, idx): # return german sentence, english sentence
        return self.de_data[idx], self.en_data[idx]

def create_batch(each_data_batch, PAD_IDX):
    de_batch, en_batch = [], []
    for (de_item, en_item) in each_data_batch:
        de_batch.append(de_item)
        en_batch.append(en_item)
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch

def build_vocab(filepath, tokenizer):
    # Build vocabulary: word -> token mapping
    counter = Counter()
    with open(filepath, encoding="utf8") as fh:
        for str in fh:
            counter.update(tokenizer(str))
    fh.close()
    return Vocab(counter=counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def tokenize(data_path, vocab, tokenizer, device):
    # tokenize sentences
    data = []
    with open(data_path, encoding="utf8") as f:
        for sent in f:
            tok_arr = [vocab['<bos>']]
            for token in tokenizer(sent.rstrip("\n")):
                tok_arr.append(vocab[token])
            tok_arr.append(vocab['<eos>'])
            data.append(torch.tensor(tok_arr, device=device))
    f.close()
    return data

def detokenize(vocab, tokens):
    # detokenize sentences: token -> word
    sent = []
    itos = vocab.itos
    for t in range(len(tokens)):
        sent.append(itos[tokens[t]])
    return sent

def train():
    train_germ_path = '../data/train/train.de'
    train_eng_path = '../data/train/train.en'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # vocab: list of word to token mappings i.e. en_vocab['what'] <-> token number     
    de_vocab = build_vocab(train_germ_path, de_tokenizer)
    en_vocab = build_vocab(train_eng_path, en_tokenizer)
    # Build tokens
    de_tokens = tokenize(data_path=train_germ_path, vocab=de_vocab, tokenizer=de_tokenizer, device=device)
    en_tokens = tokenize(data_path=train_eng_path, vocab=en_vocab, tokenizer=en_tokenizer, device=device)
    # dataset
    train_data = MyDataset(de_tokenized=de_tokens, en_tokenized=en_tokens)
    batch_size_t = 50
    # create batches of sample for training
    train_iter = torch.utils.data.DataLoader(
                            train_data,
                            batch_size=batch_size_t,
                            shuffle=True,
                            collate_fn= lambda batch: create_batch(each_data_batch=batch, PAD_IDX=de_vocab['<pad>']))

    de_size = de_vocab.__len__()
    en_size = en_vocab.__len__()
    # declare model
    model = MyTransformer(num_encoder_layers=3,
                          num_decoder_layers=3,
                          emb_size=512,
                          dim_feedforward=512,
                          nhead=8,
                          src_vocab_size=de_size,
                          tgt_vocab_size=en_size).to(device)
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=de_vocab['<pad>'])
    epochs = 20

    for epoch in range(epochs):
        avg_loss_train = 0
        # start training
        for de_sent, en_sent in train_iter:
            tgt_input = en_sent[:-1]
            tgt_out = en_sent[1:]
            src = de_sent
            optimizer.zero_grad()
            # forward prop
            logits = model(src=src, trg=tgt_input, memory_key_padding_mask=None, PAD_IDX=torch.tensor(de_vocab['<pad>'], device=device))
            # calculate loss
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            avg_loss_train += loss.item()
            # back prop
            loss.backward()
            optimizer.step()

        print(f'{avg_loss_train/(train_data.__len__()/batch_size_t)}')
    # save model
    torch.save(model.state_dict(), f'../utils/model.pt')
    return None

def test():
    model_path = '../utils/model.pt'
    de_test = '../data/test/test.de'
    en_test = '../data/test/test.en'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # load saved vocab
    de_vocab = torch.load('../utils/de_vocab.pth')
    en_vocab = torch.load('../utils/en_vocab.pth')
    # get tokens
    de_tokens = tokenize(data_path=de_test, vocab=de_vocab, tokenizer=de_tokenizer, device=device)
    en_tokens = tokenize(data_path=en_test, vocab=en_vocab, tokenizer=en_tokenizer, device=device)
    # data
    model_test = MyTransformer(num_encoder_layers=3,
                          num_decoder_layers=3,
                          emb_size=512,
                          dim_feedforward=512,
                          nhead=8,
                          src_vocab_size=de_vocab.__len__(),
                          tgt_vocab_size=en_vocab.__len__()).to(device)
    model_test.load_state_dict(torch.load(model_path))
    model_test.eval()
    test_data = MyDataset(de_tokenized=de_tokens, en_tokenized=en_tokens)

    with open('../out.txt', 'w') as f:
        for s in range(test_data.__len__()):
            de_sent, en_sent = test_data.__getitem__(s)
            pred = torch.tensor([[en_vocab['<bos>']]]).to(device)
            for t in range(len(en_sent)):
                # get predictions
                logits = model_test(src=de_sent.unsqueeze(-1),
                                    trg=pred, 
                                    memory_key_padding_mask=None, 
                                    PAD_IDX=torch.tensor(de_vocab['<pad>'], 
                                    device=device))
                # append predicted word (argmax last logit) to prediction sentence
                pred = torch.cat((pred, logits[-1].argmax().unsqueeze(0).unsqueeze(0)), dim=0)
                # Stop at <eos> (end of sentence)
                if logits[-1].argmax().item() == en_vocab['<eos>']: 
                    break
            # get words from tokens
            input = detokenize(de_vocab, de_sent.tolist())
            target = detokenize(en_vocab, en_sent.tolist())
            predict = detokenize(en_vocab, pred.squeeze(-1).tolist())

            f.write(f'Input: {" ".join(input[1:len(input) - 1])}\n')
            f.write(f'Target: {" ".join(target[1:len(target) - 1])}\n')
            f.write(f'Prediction: {" ".join(predict[1:len(predict) - 1])}\n\n')
    f.close()

if __name__ == "__main__":
    # train() # train model 
    test()