import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from flask import Flask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

def load_model(model_path):
    model = SentimentClassifier(n_classes=7)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict_emotion(model, comment):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    encoding = tokenizer.encode_plus(
        comment,
        truncation=True,
        add_special_tokens=True,
        max_length=120,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    _, prediction = torch.max(output, dim=1)
    emotion_mapping = {0: 'Enjoyment', 1: 'Disgust', 2: 'Sadness', 3: 'Anger', 4: 'Surprise', 5: 'Fear', 6: 'Other'}
    emotion = emotion_mapping[prediction.item()]

    return emotion
