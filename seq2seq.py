import pickle
import os
from sys import get_asyncgen_hooks
from numpy.core import numeric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.onnx import _optimize_trace
import torch.optim as optim

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

class Encoder(nn.Module):
    """Some Information about Encoder (bi-directional)"""
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, num_layers=6, padding_idx=3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        _, state = self.lstm(x)
        h, c = state
        h = torch.sum(torch.stack(torch.split(h, 2, dim=0)), 1, keepdim=False)
        c = torch.sum(torch.stack(torch.split(c, 2, dim=0)), 1, keepdim=False)
        return (h, c)

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, num_layers=6, padding_idx=3):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state):
        x = self.embedding(x)
        output, state = self.lstm(x, state)
        output = self.linear(output)
        #output = F.softmax(output, dim=2)
        return output, state

class Seq2SeqModule(nn.Module):
    """Some Information about Seq2SeqModule"""
    def __init__(self, vocab_size, embedding_dim=200, encoder_hidden_dim=256, decoder_hidden_dim=256, padding_idx=3, num_layers=6):
        super(Seq2SeqModule, self).__init__()
        self.encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=encoder_hidden_dim, padding_idx=padding_idx, num_layers=num_layers)
        self.decoder = Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=decoder_hidden_dim, padding_idx=padding_idx, num_layers=num_layers)

    def forward(self, x, src):
        state = self.encoder(x)
        decoder_output, _ = self.decoder(src, state)
        return decoder_output

class Seq2SeqResponder:
    def __init__(self, vocab_size=8000, sentence_length=30, embedding_dim=200, encoder_hidden_dim=256, decoder_hidden_dim=256, padding_idx=3, num_layers=6):
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.padding_idx = padding_idx
        self.s2s = Seq2SeqModule(vocab_size, embedding_dim=embedding_dim, encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim, padding_idx=padding_idx, num_layers=num_layers)

    def train_sentencepiece(self, src_file_path, sp_model_prefix):
        spm.SentencePieceTrainer.Train(
            f"--input={src_file_path}, --model_prefix={sp_model_prefix} --character_coverage=0.9995 --vocab_size={self.vocab_size} --pad_id=3"
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{sp_model_prefix}.model")

    def train_seq2seq(self, src_file_path, steps=50, num_epoch=10, batch_size=100, save_path="seq2seq.pkl"):
        # load data
        with open(src_file_path, "r") as f:
            data = f.read()

        data = data.split('\n') # 改行で区切る
        data = np.array([self.sentence_to_id(s, self.sentence_length) for s in tqdm(data)]) # パース処理

        input_data = data[0:-2]
        output_data = data[1:-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s2s.to(device)
        optimizer = optim.AdamW(self.s2s.parameters(), lr=0.0003)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        for step in range(0, steps):
            print("="*60)
            print(f"STEP # {step+1} / {steps}")
            print("="*60)
            for epoch in range(0, num_epoch):
                epoch_loss = 0
                input_batch, output_batch = train2batch(input_data, output_data, batch_size=batch_size)
                for i in tqdm(range(0, len(input_batch))):
                    self.s2s.train()
                    loss = 0
                    batch_loss = 0
                    input_tensor = torch.LongTensor(input_batch[i]).to(device)
                    tgt_tensor = torch.LongTensor(output_batch[i]).to(device)
                    self.s2s.zero_grad()
                    src_tensor = self.tgt2src(tgt_tensor)
                    output = self.s2s(input_tensor, src_tensor)
                    for j in range(0, output.size(1)):
                        loss += criterion(output[:, j, :], tgt_tensor[:, j])
                        batch_loss += loss.item()
                    batch_loss /= output.size(1)
                    epoch_loss += batch_loss
                    loss.backward()
                    optimizer.step()
                    #print(f"BATCH #{i} / {len(input_batch)} loss: {batch_loss}")
                print("-"*60)
                print(f"EPOCH #{epoch+1} / {num_epoch} avg.loss: {epoch_loss/len(input_batch)}")
                sample_input = "おはようございます"
                sample_output = self.predict_from_sentences([sample_input])[0]
                print(f"sentence sample: {sample_output}")
                print("-"*60)
            scheduler.step()
            self.save(save_path)
                
    
    def sentence_to_id(self, sentence, length):
        r = self.sp.EncodeAsIds(sentence)[0:length-1]
        while len(r) < length:
            r.append(self.padding_idx)
        return r
    
    def train(self, src_file_path, sp_model_prefix="sentencepiece", save_path="seq2seq.pkl", steps=100, num_epoch=10, batch_size=100):
        if os.path.exists(f"{sp_model_prefix}.model"):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(f"{sp_model_prefix}.model")
        else:
            self.train_sentencepiece(src_file_path, sp_model_prefix)
        self.train_seq2seq(src_file_path, steps=steps, num_epoch=num_epoch, batch_size=batch_size)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def tgt2src(self, output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.cat((torch.full((output.size(0), 1), self.padding_idx).to(device), output), axis=1)
        return t[:, :-1] # right shift

    def predict_with_batch(self, input_data, batch_size=500, noise_gain=0.0, flag_gpu = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not flag_gpu:
            device = torch.device("cpu")
        self.s2s.eval()
        self.s2s.to(device)
        input_batch = test2batch(input_data, batch_size=batch_size)
        results = []
        with torch.no_grad():
            for i in range(0, len(input_batch)):
                batch_tmp = []
                input_tensor = torch.LongTensor(input_batch[i]).to(device)
                src_tensor = torch.full((input_tensor.size(0), 1), self.padding_idx+5).to(device)
                state = self.s2s.encoder(input_tensor)
                h, c = state
                h += (torch.randn(h.size()).to(device) - 0.5) * noise_gain
                c += (torch.randn(c.size()).to(device) - 0.5) * noise_gain
                state = (h, c)
                decoder_hidden = state
                for j in range(0, input_tensor.size(1)):
                    decoder_output, decoder_hidden = self.s2s.decoder(src_tensor, decoder_hidden)
                    decoder_output = torch.argmax(decoder_output, dim=2)
                    src_tensor = decoder_output
                    batch_tmp.append(decoder_output)
                batch_tmp = torch.cat(batch_tmp, dim=1)
                results.append(batch_tmp)
            results = torch.cat(results)
        return results

    def predict_from_sentences(self, sentences, batch_size=500, noise_gain=0.0, flag_gpu=True):
        input_data = np.array([self.sentence_to_id(s, self.sentence_length) for s in sentences]) 
        results = self.predict_with_batch(input_data, batch_size=batch_size, noise_gain=noise_gain, flag_gpu=flag_gpu).tolist()
        return [ self.sp.DecodeIdsWithCheck(s) for s in results ]
    
    @classmethod
    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

# utilites
def train2batch(x, y, batch_size=100):
    rx, ry = [], []
    for i in range(0, len(x), batch_size):
        rx.append(x[i:i+batch_size])
        ry.append(y[i:i+batch_size])
    return rx, ry

def test2batch(x, batch_size=100):
    rx = []
    for i in range(0, len(x), batch_size):
        rx.append(x[i:i+batch_size])
    return rx