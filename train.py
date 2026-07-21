import argparse
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_NAME = "distilbert-base-uncased"


# Data utilities
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)          # removing whitespace/newlines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # droping non-ASCII artifacts
    return text.strip()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df["Resume_str"].apply(clean_text)
    df = df[df["text"].str.len() > 20].reset_index(drop=True)
    return df[["text", "Category"]]


class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Train / eval
def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            if train:
                optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="Resume/Resume.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="resume_classifier_model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data(args.csv_path)
    print(f"Total resumes: {len(df)} | Categories: {df['Category'].nunique()}")

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Category"])
    num_labels = len(label_encoder.classes_)

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    train_ds = ResumeDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len)
    val_ds = ResumeDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_len)
    test_ds = ResumeDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, val_loader, optimizer, scheduler, device, train=False
        )
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            np.save(f"{args.output_dir}/label_classes.npy", label_encoder.classes_)

    # Final test evaluation using the best checkpoint 
    model = DistilBertForSequenceClassification.from_pretrained(args.output_dir).to(device)
    _, test_acc, test_preds, test_labels = run_epoch(model, test_loader, optimizer, scheduler, device, train=False)
    test_f1 = f1_score(test_labels, test_preds, average="macro")

    print("\nTest set results")
    print(f"Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
    print(
        classification_report(
            test_labels, test_preds, target_names=label_encoder.classes_, zero_division=0
        )
    )
    print(f"\nSaved fine-tuned model + tokenizer to {args.output_dir}/")


if __name__ == "__main__":
    main()
