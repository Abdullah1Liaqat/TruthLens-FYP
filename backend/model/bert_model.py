import torch.nn as nn
from transformers import BertModel, BertConfig


class BertBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.hidden_dropout_prob = 0.1
        config.num_labels = 1

        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)
