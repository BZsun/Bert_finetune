import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import (
    BertTokenizer, 
    BertForTokenClassification
    )
from torch.utils.data import (
    Dataset,
    DataLoader, 
    TensorDataset
    )
from network import BERTRelationEtract
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
    )
from tqdm import tqdm
from ipdb import set_trace
import json
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class REDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_length):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.convert_tokens_to_ids(self.inputs[idx][0])
        attention_mask = [1] * len(input_ids)
        labels = [self.inputs[idx][1]]

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id]*padding_length
            attention_mask = attention_mask + [0]*padding_length
        else:
            input_ids = input_ids[: self.max_length]
        token_type_ids = [0] * len(input_ids) 

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'token_type_ids': torch.LongTensor(token_type_ids),
            'labels': torch.LongTensor(labels),
        }

class ReFinetune:
    def __init__(self):
        self.pretrained_path = '/mnt/g/pretrained/bert-base-uncased'
        self.finetuned_model = '/mnt/g/my_finetuned/re/re_model.pt'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.max_length = 128
        self.batch_size = 8
        self.lr = 2e-5
        self.epochs = 3
        self.trainfile = './fewrel/fewrel_train.txt'
        self.jsonfile = './fewrel/fewrel_train_rel2id.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.jsonfile, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        self.id2label = {v: k for k,v in self.label2id.items()}
        self.num_labels = len(self.label2id)

    def mark_entity_token(self, line):
        text = line["token"]
        head = line["h"]
        tail = line["t"]
        label = line["relation"]
        text[head["pos"][0]] = '<head>' + text[head["pos"][0]]
        text[head["pos"][1]-1] = text[head["pos"][1]-1] + '</head>'

        text[tail["pos"][0]] = '<tail>' + text[tail["pos"][0]]
        text[tail["pos"][1]-1] = text[tail["pos"][1]-1] + '</tail>'
        label = line['relation']

        return text, label

    def read_data(self):
        sents, labels = [], []
        with open(self.trainfile, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                text, label = self.mark_entity_token(line)
                sents.append(text)
                labels.append(label)

        return sents, labels
        
    def gen_inputs(self, sentences, labels):
        inputs = []
        for s_idx in range(len(sentences)):
            sent = sentences[s_idx]
            label = labels[s_idx]
            word_pieces = []
            labels_pieces = self.label2id[label]
            for i, word in enumerate(sent):
                wp = self.tokenizer.tokenize(word)
                word_pieces.extend(wp)
            inputs.append((word_pieces, labels_pieces))

        return inputs

    def load_dataloader(self):
        sents, labels = self.read_data()
        inputs = self.gen_inputs(sents, labels)
        length = int(len(inputs) * 0.8)
        inputs_train = inputs[:length]
        inputs_valid = inputs[length:]
        train_set = REDataset(inputs_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        valid_set = REDataset(inputs_valid, self.tokenizer, self.max_length)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True)

        return train_loader, valid_loader

    def train_model(self, model, train_loader, valid_loader):
        model.train()
        optimizer = AdamW(model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        best_acc = 0
        for epoch in range(self.epochs):
            train_loss = 0
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids = token_type_ids,
                    labels=labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            _acc = self.eval_model(model, valid_loader)
            
            if _acc > best_acc:
                best_acc = _acc
                torch.save(model.state_dict(), self.finetuned_model)
                print('===model has saved to {}==='.format(self.finetuned_model))

            print('Epoch: {}\nAverage Training Loss: {}'.format(epoch+1, train_loss))

    def eval_model(self, model, valid_loader):
        dev_acc = 0.0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids = token_type_ids
                )
                y_true = labels.cpu().numpy()
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                dev_acc += accuracy_score(y_true, y_pred)
        dev_acc /= len(valid_loader)

        return dev_acc

    def predict(self):
        model = BERTRelationEtract(self.num_labels, self.pretrained_path).to(self.device)
        model.load_state_dict(torch.load(self.finetuned_model))
        model.to(self.device)
        model.eval()  

    def main(self):
        train_loader, valid_loader = self.load_dataloader()
        model = BERTRelationEtract(self.num_labels, self.pretrained_path).to(self.device)
        self.train_model(model, train_loader, valid_loader)

if __name__ == '__main__':
    opt = ReFinetune()
    opt.main()
