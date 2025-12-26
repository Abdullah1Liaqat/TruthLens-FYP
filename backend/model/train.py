import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.bert_model import BertBinaryClassifier
from utils.preprocessing import load_liar_dataset, tokenize_texts


class LiarDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def train_model(data_path, epochs=3, batch_size=16, lr=2e-5, save_path="bert_liar.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df, _ = load_liar_dataset(data_path)

    X_train, y_train = train_df["statement"].tolist(), train_df["label_num"].tolist()
    X_valid, y_valid = valid_df["statement"].tolist(), valid_df["label_num"].tolist()

    train_enc = tokenize_texts(X_train)
    valid_enc = tokenize_texts(X_valid)

    train_ds = LiarDataset(train_enc, y_train)
    valid_ds = LiarDataset(valid_enc, y_valid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    model = BertBinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
