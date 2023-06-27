import torch.nn as nn
from transformers import BertModel

class BERTRelationEtract(nn.Module):
    def __init__(self, num_labels, pretrained_path):
        super(BERTRelationEtract, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.3)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits