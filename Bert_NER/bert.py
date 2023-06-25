import torch
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
from network import BertCRF
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from ipdb import set_trace

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class NERDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_length):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.convert_tokens_to_ids(self.inputs[idx][0])
        attention_mask = [1] * len(input_ids)
        labels = [label2id[lab] for lab in self.inputs[idx][1]]

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id]*padding_length
        attention_mask = attention_mask + [0]*padding_length
        token_type_ids = [0] * len(input_ids) 
        labels = labels + [1] * padding_length

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'token_type_ids': torch.LongTensor(token_type_ids),
            'labels': torch.LongTensor(labels),
        }

class MyFinetune:
    def __init__(self, label2id):
        self.NUM_LABELS = len(label2id)
        self.MAX_LEN = 180
        self.batch_size = 4
        self.epochs = 1
        self.lr = 2e-5
        self.pretrained_path = '/mnt/g/pretrained/bert-base-uncased'
        self.finetuned_model = '/mnt/g/my_finetuned/ner/ner_model.pt'
        # self.dnn = BertForTokenClassification.from_pretrained(self.pretrained_path, num_labels=self.NUM_LABELS)
        self.dnn = BertCRF(self.pretrained_path, num_tags=self.NUM_LABELS)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def read_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        sentences = []
        labels = []
        sent = []
        label = []
        for line in lines:
            if line == '\n':
                sentences.append(sent)
                labels.append(label)
                sent = []
                label = []
            else:
                token, _, _, tag = line.split()
                sent.append(token)
                label.append(tag)

        return sentences, labels

    def gen_inputs(self, filename):
        sentences, labels = self.read_data(filename)
        inputs = []
        for s_idx in range(len(sentences)):
            sent = sentences[s_idx]
            label = labels[s_idx]
            word_pieces = []
            labels_pieces = []
            for i, word in enumerate(sent):
                wp = self.tokenizer.tokenize(word)
                word_pieces.extend(wp)
                if label[i] != 'O':
                    labels_pieces.extend([label[i]] + ['I-'+label[i][2:]]*(len(wp)-1))
                else:
                    labels_pieces.extend(['O']*len(wp))
            inputs.append((word_pieces, labels_pieces))

        return inputs

    def gen_dataloader(self, filename, shuffle):
        inputs = self.gen_inputs(filename)
        data_set = NERDataset(inputs, self.tokenizer, self.MAX_LEN)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=shuffle)

        return data_loader

    def train_model(self, train_loader, valid_loader):
        model = self.dnn
        model.to(self.device)
        model.train()
        total_steps = len(train_loader) * self.epochs
        optimizer = AdamW(model.parameters(), lr=self.lr)
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        best_loss = 100
        for epoch in range(self.epochs):
            total_loss = 0
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
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            _loss = self.eval_model(model, valid_loader)
            if _loss < best_loss:
                best_loss = _loss
                torch.save(model.state_dict(), self.finetuned_model)

            print('Epoch: {}\nAverage Training Loss: {}\n Average Validation Loss: {}'.format(
                epoch+1, total_loss/len(train_loader), _loss))

    def eval_model(self, model, eval_loader):
        model = model.to(self.device)
        model.eval()
        total_loss = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids = token_type_ids,
                    labels=labels
                )
                total_loss += loss.item()
                total += len(labels)
            _loss = total_loss / total

        return _loss

    def main(self):
        train_loader = self.gen_dataloader('./conll2003/train.txt', shuffle=True)
        valid_loader = self.gen_dataloader('./conll2003/valid.txt', shuffle=True)
        self.train_model(train_loader, valid_loader)

    def predict(self, id2label):
        res = []
        model = self.dnn
        # model.load_state_dict(torch.load(finetune_model, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(self.finetuned_model))
        model.to(self.device)
        model.eval()
        test_loader = self.gen_dataloader('./conll2003/test.txt', shuffle=False)
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids = token_type_ids
                )
                inp_texts = [self.tokenizer.convert_ids_to_tokens(i.tolist()) for i in input_ids]
                inp_texts = [[j for j in i if j != '[PAD]'] for i in inp_texts]
                pred_texts = [[id2label[j] for j in i] for i in preds]
                set_trace()
                out = [[x, y] for x, y in zip(inp_texts, pred_texts)]
                res.extend(out)
        return res


if __name__ == '__main__':
    label2id = {
                'UNK': 0, 'PAD':1, 'O': 2, 
                'B-LOC': 3, 'I-LOC': 4,
                'B-PER': 5, 'I-PER': 6, 
                'B-ORG': 7,'I-ORG': 8, 
                'B-MISC': 9, 'I-MISC': 10
                }
    id2label = {v: k for k, v in label2id.items()}
    opt = MyFinetune(label2id)
    opt.main()
    opt.predict(id2label)

