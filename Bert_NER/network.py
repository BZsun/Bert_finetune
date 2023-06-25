import torch
import torch.nn.functional as F
from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel

class BertCRF(nn.Module):
    def __init__(self, pretrained_path, num_tags):
        super(BertCRF, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.crf = CRF(num_tags, batch_first=True)
        self.classifier = nn.Linear(768, num_tags)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = self.dropout(logits)
        if labels is not None:
            loss = self.crf(logits, labels, mask=attention_mask.bool())
            return -loss
        else:
            predicted_tags = self.crf.decode(logits, mask=attention_mask.bool())
            return predicted_tags