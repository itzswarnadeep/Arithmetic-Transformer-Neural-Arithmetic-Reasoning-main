#!/usr/bin/env python3
"""
Arithmetic Transformer — Full Training Pipeline
=================================================
Trains an encoder–decoder Transformer to learn addition and subtraction.
Optimized with MPS support for Mac acceleration.
"""

import os
import sys
import math
import time
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — use local working directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "training_set.json")
TEST_PATH  = os.path.join(BASE_DIR, "test_set.json")
VAL_PATH   = os.path.join(BASE_DIR, "validation_set.json")
GEN_PATH   = os.path.join(BASE_DIR, "generalization_set.json")

# Determine device (CUDA -> MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Data Generation (only if files do not exist)
# ---------------------------------------------------------------------------

def create_arithmetic_examples(
    example_count,
    allowed_operators=['+', '-'],
    operand_min_digits=1,
    operand_max_digits=5,
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    examples = []
    for _ in range(example_count):
        operator = random.choice(allowed_operators)
        first_num_digits = random.randint(operand_min_digits, operand_max_digits)
        second_num_digits = random.randint(operand_min_digits, operand_max_digits)
        first_num = random.randint(10**(first_num_digits-1), 10**first_num_digits - 1)
        second_num = random.randint(10**(second_num_digits-1), 10**second_num_digits - 1)

        if operator == '-' and first_num < second_num:
            first_num, second_num = second_num, first_num

        problem = f"{first_num}{operator}{second_num}"
        if operator == '+':
            answer = str(first_num + second_num)
        else:
            answer = str(first_num - second_num)

        examples.append((problem, answer))

    return examples


def create_dataset_with_special_cases(
    standard_example_count,
    special_case_count,
    allowed_operators=['+', '-'],
    operand_min_digits=1,
    operand_max_digits=5,
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    standard_examples = create_arithmetic_examples(
        example_count=standard_example_count,
        allowed_operators=allowed_operators,
        operand_min_digits=operand_min_digits,
        operand_max_digits=operand_max_digits,
        random_seed=random_seed
    )

    special_cases = []

    carrying_count = int(0.3 * special_case_count)
    for _ in range(carrying_count):
        num1 = random.randint(900, 999)
        num2 = random.randint(900, 999)
        problem = f"{num1}+{num2}"
        answer = str(num1 + num2)
        special_cases.append((problem, answer))

    borrowing_count = int(0.3 * special_case_count)
    for _ in range(borrowing_count):
        num1 = random.randint(1000, 9999)
        num2 = random.randint(1, 999)
        while not any(int(d1) < int(d2) for d1, d2 in zip(str(num1).zfill(4)[::-1], str(num2).zfill(4)[::-1])):
            num1 = random.randint(1000, 9999)
            num2 = random.randint(1, 999)
        problem = f"{num1}-{num2}"
        answer = str(num1 - num2)
        special_cases.append((problem, answer))

    leading_zeros_count = int(0.2 * special_case_count)
    for _ in range(leading_zeros_count):
        num1 = random.randint(1000, 9999)
        num2 = random.randint(900, num1 - 100)
        problem = f"{num1}-{num2}"
        answer = str(num1 - num2)
        special_cases.append((problem, answer))

    repeated_digits_count = int(0.2 * special_case_count)
    for _ in range(repeated_digits_count):
        digit = random.randint(1, 9)
        num1 = int(str(digit) * random.randint(3, 5))
        num2 = int(str(digit) * random.randint(3, 5))
        problem = f"{num1}+{num2}"
        answer = str(num1 + num2)
        special_cases.append((problem, answer))

    complete_dataset = standard_examples + special_cases
    random.shuffle(complete_dataset)

    return complete_dataset

# ---------------------------------------------------------------------------
# 2. Load / generate datasets
# ---------------------------------------------------------------------------

def generate_datasets_if_needed():
    if all(os.path.exists(p) for p in [TRAIN_PATH, TEST_PATH, VAL_PATH, GEN_PATH]):
        print("Dataset JSON files already exist — skipping generation.")
        return

    print("Generating datasets…")
    training_set = create_dataset_with_special_cases(75000, 5000, random_seed=42)
    validation_set = create_dataset_with_special_cases(9000, 1000, random_seed=43)
    test_set = create_dataset_with_special_cases(9000, 1000, random_seed=44)
    generalization_set = create_dataset_with_special_cases(
        4500, 500, operand_min_digits=6, operand_max_digits=8, random_seed=45
    )

    for path, data in [
        (TRAIN_PATH, training_set),
        (VAL_PATH, validation_set),
        (TEST_PATH, test_set),
        (GEN_PATH, generalization_set),
    ]:
        with open(path, "w") as f:
            json.dump(data, f)

    print(f"Training set size: {len(training_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Generalization set size: {len(generalization_set)}")


def load_math_datasets():
    with open(TRAIN_PATH) as f:
        train_list = json.load(f)
    with open(TEST_PATH) as f:
        test_list = json.load(f)
    with open(VAL_PATH) as f:
        val_list = json.load(f)
    with open(GEN_PATH) as f:
        gen_list = json.load(f)
    return train_list, test_list, val_list, gen_list

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

MAX_LEN = 20
D_MODEL = 512

def preprocess_data(data, max_input_length=MAX_LEN, max_output_length=MAX_LEN):
    vocab = "0123456789+-"
    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    char_to_idx["<pad>"] = 0
    char_to_idx["<sos>"] = len(char_to_idx)
    char_to_idx["<eos>"] = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    input_sequences = []
    output_sequences = []

    for problem, solution in data:
        input_seq = [char_to_idx[char] for char in problem]
        output_seq = [char_to_idx[char] for char in solution]

        input_seq = [char_to_idx["<sos>"]] + input_seq[:max_input_length - 1]
        input_seq = input_seq + [char_to_idx["<eos>"]] + [char_to_idx["<pad>"]] * (max_input_length - len(input_seq) - 1)

        output_seq = [char_to_idx["<sos>"]] + output_seq[:max_output_length - 1]
        output_seq = output_seq + [char_to_idx["<eos>"]] + [char_to_idx["<pad>"]] * (max_output_length - len(output_seq) - 1)

        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

    return np.array(input_sequences), np.array(output_sequences), char_to_idx, idx_to_char

# ---------------------------------------------------------------------------
# 4. Transformer Model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=MAX_LEN):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x_t = x.transpose(0, 1)
        attn_output, _ = self.self_attn(x_t, x_t, x_t, key_padding_mask=key_padding_mask, need_weights=False)
        attn_output = attn_output.transpose(0, 1)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x_t = x.transpose(0, 1)
        enc_output_t = enc_output.transpose(0, 1)
        self_attn_output, _ = self.self_attn(x_t, x_t, x_t, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        self_attn_output = self_attn_output.transpose(0, 1)
        x = self.norm1(x + self.dropout(self_attn_output))
        x_t = x.transpose(0, 1)
        cross_attn_output, _ = self.cross_attn(x_t, enc_output_t, enc_output_t, key_padding_mask=memory_key_padding_mask, need_weights=False)
        cross_attn_output = cross_attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=MAX_LEN):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=100):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return x


class ArithmeticTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=8, d_ff=512, num_layers=3, dropout=0.1, max_len=20):
        super(ArithmeticTransformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_masks(self, src, tgt):
        src_pad_mask = (src == 0).to(src.device)
        tgt_pad_mask = (tgt == 0).to(tgt.device)
        tgt_len = tgt.size(1)
        subsequent_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)
        return src_pad_mask, tgt_pad_mask, subsequent_mask

    def forward(self, src, tgt):
        src_pad_mask, tgt_pad_mask, tgt_subsequent_mask = self.create_masks(src, tgt)
        enc_output = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask=tgt_subsequent_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.output_projection(dec_output)
        return output

    def generate(self, src, max_len=20, sos_token_id=13, eos_token_id=14, pad_token_id=0):
        batch_size = src.size(0)
        device = src.device
        output = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        src_pad_mask = (src == pad_token_id).to(device)
        enc_output = self.encoder(src, src_key_padding_mask=src_pad_mask)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_len = output.size(1)
            tgt_subsequent_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
            tgt_pad_mask = (output == pad_token_id).to(device)
            dec_output = self.decoder(output, enc_output, tgt_mask=tgt_subsequent_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
            logits = self.output_projection(dec_output[:, -1])
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            next_token[finished] = pad_token_id
            output = torch.cat([output, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        if output.size(1) < max_len:
            pad_len = max_len - output.size(1)
            padding = torch.full((batch_size, pad_len), pad_token_id, dtype=torch.long, device=device)
            output = torch.cat([output, padding], dim=1)

        return output

# ---------------------------------------------------------------------------
# 5. Training helpers
# ---------------------------------------------------------------------------

def train_model(model, train_dataloader, val_dataloader, config):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    os.makedirs("checkpoints", exist_ok=True)

    history = defaultdict(list)
    best_val_accuracy = 0.0

    for epoch in range(config["epochs"]):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for src, tgt in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            optimizer.zero_grad()
            output = model(src, tgt_inp)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_out = tgt_out.contiguous().view(-1)
            loss = criterion(output, tgt_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate_model(model, val_dataloader, criterion)
        history["val_loss"].append(val_metrics["loss"])
        history["val_exact_match"].append(val_metrics["exact_match"])
        history["val_digit_accuracy"].append(val_metrics["digit_accuracy"])
        history["perplexity"].append(val_metrics["perplexity"])

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config['epochs']} — "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Exact Match: {val_metrics['exact_match']:.4f}, "
            f"Val Digit Accuracy: {val_metrics['digit_accuracy']:.4f}, "
            f"Perplexity: {val_metrics['perplexity']:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        if val_metrics["exact_match"] > best_val_accuracy:
            best_val_accuracy = val_metrics["exact_match"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config, "checkpoints/best_model.pth")
            print(f"  → New best model saved (val accuracy: {best_val_accuracy:.4f})")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, config, f"checkpoints/model_epoch_{epoch+1}.pth")

    return history


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    all_exact_matches = []
    all_digit_accuracies = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            output = model(src, tgt_inp)
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_out_flat = tgt_out.contiguous().view(-1)
            loss = criterion(output_flat, tgt_out_flat)
            total_loss += loss.item()
            num_batches += 1
            total_tokens += (tgt_out_flat != 0).sum().item()

            predictions = model.generate(src)

            for i in range(len(src)):
                true_seq = tgt[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()
                true_seq = true_seq[true_seq != 0]
                pred_seq = pred_seq[pred_seq != 0]
                true_seq = true_seq[(true_seq != 14) & (true_seq != 13)]
                pred_seq = pred_seq[(pred_seq != 14) & (pred_seq != 13)]

                exact_match = 1 if np.array_equal(true_seq, pred_seq) else 0
                all_exact_matches.append(exact_match)

                min_len = min(len(true_seq), len(pred_seq))
                correct_digits = sum(1 for j in range(min_len) if true_seq[j] == pred_seq[j])
                digit_accuracy = correct_digits / max(len(true_seq), len(pred_seq)) if max(len(true_seq), len(pred_seq)) > 0 else 0
                all_digit_accuracies.append(digit_accuracy)

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    avg_exact_match = sum(all_exact_matches) / max(len(all_exact_matches), 1)
    avg_digit_accuracy = sum(all_digit_accuracies) / max(len(all_digit_accuracies), 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "exact_match": avg_exact_match,
        "digit_accuracy": avg_digit_accuracy,
    }


def test_model(model, test_dataloader, generalization_dataloader):
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Evaluating on test set…")
    test_metrics = evaluate_model(model, test_dataloader, criterion)
    print("Evaluating on generalization set…")
    gen_metrics = evaluate_model(model, generalization_dataloader, criterion)

    print("\nTest Set Results:")
    print(f"  Exact Match Accuracy: {test_metrics['exact_match']:.4f}")
    print(f"  Digit-level Accuracy: {test_metrics['digit_accuracy']:.4f}")
    print(f"  Perplexity: {test_metrics['perplexity']:.4f}")

    print("\nGeneralization Set Results:")
    print(f"  Exact Match Accuracy: {gen_metrics['exact_match']:.4f}")
    print(f"  Digit-level Accuracy: {gen_metrics['digit_accuracy']:.4f}")
    print(f"  Perplexity: {gen_metrics['perplexity']:.4f}")

    return {"test": test_metrics, "generalization": gen_metrics}


def save_checkpoint(model, optimizer, epoch, metrics, config, filepath):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_training_metrics(history):
    sns.set_theme()
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history["train_loss"], color="#2ecc71", label="Training Loss", linewidth=2)
    ax1.plot(history["val_loss"], color="#e74c3c", label="Validation Loss", linewidth=2)
    ax1.set_title("Training and Validation Loss Over Time", pad=15, fontsize=12)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history["val_exact_match"], color="#3498db", linewidth=2)
    ax2.set_title("Exact Match Accuracy", pad=10, fontsize=12)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.7)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(history["val_digit_accuracy"], color="#9b59b6", linewidth=2)
    ax3.set_title("Digit-level Accuracy", pad=10, fontsize=12)
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy")
    ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300, bbox_inches="tight")
    print("Saved training_metrics.png")
    plt.close()


def prepare_dataloaders(dataset_dict, batch_size=128):
    train_dataset = TensorDataset(torch.LongTensor(dataset_dict["X_train"]), torch.LongTensor(dataset_dict["y_train"]))
    val_dataset = TensorDataset(torch.LongTensor(dataset_dict["X_val"]), torch.LongTensor(dataset_dict["y_val"]))
    test_dataset = TensorDataset(torch.LongTensor(dataset_dict["X_test"]), torch.LongTensor(dataset_dict["y_test"]))
    gen_dataset = TensorDataset(torch.LongTensor(dataset_dict["X_gen"]), torch.LongTensor(dataset_dict["y_gen"]))

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
        "generalization": DataLoader(gen_dataset, batch_size=batch_size),
    }


def analyze_errors(model, dataloader, idx_to_char, num_examples=10):
    model.eval()
    errors = []
    correct = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            predictions = model.generate(src)

            for i in range(len(src)):
                input_seq = src[i].cpu().numpy()
                true_seq = tgt[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()

                input_seq = input_seq[input_seq != 0]
                true_seq = true_seq[true_seq != 0]
                pred_seq = pred_seq[pred_seq != 0]

                input_str = "".join([idx_to_char[idx] for idx in input_seq])
                true_str = "".join([idx_to_char[idx] for idx in true_seq])
                pred_str = "".join([idx_to_char[idx] for idx in pred_seq])

                example = {"input": input_str, "true": true_str, "pred": pred_str, "correct": true_str == pred_str}
                if example["correct"]:
                    correct.append(example)
                else:
                    errors.append(example)

            if len(errors) >= num_examples and len(correct) >= num_examples:
                break

    print("\n--- CORRECT EXAMPLES ---")
    for i, example in enumerate(random.sample(correct, min(num_examples, len(correct)))):
        print(f"{i+1}. Input: {example['input']}, True: {example['true']}, Pred: {example['pred']}")

    print("\n--- ERROR EXAMPLES ---")
    for i, example in enumerate(random.sample(errors, min(num_examples, len(errors)))):
        print(f"{i+1}. Input: {example['input']}, True: {example['true']}, Pred: {example['pred']}")

    if errors:
        addition_errors = [e for e in errors if "+" in e["input"]]
        subtraction_errors = [e for e in errors if "-" in e["input"]]
        print(f"\nTotal errors: {len(errors)}")
        print(f"Addition errors: {len(addition_errors)} ({len(addition_errors)/len(errors)*100:.2f}%)")
        print(f"Subtraction errors: {len(subtraction_errors)} ({len(subtraction_errors)/len(errors)*100:.2f}%)")


# ---------------------------------------------------------------------------
# 6. Ablation studies
# ---------------------------------------------------------------------------

def execute_ablation_studies(dataset_dict, base_config, ablation_configs):
    dataloaders = prepare_dataloaders(dataset_dict, base_config["batch_size"])
    ablation_results = {}

    print("Starting ablation studies…")
    print(f"Base configuration: {base_config}")

    for config_name, config_changes in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"Running ablation test: {config_name}")
        print(f"Configuration changes: {config_changes}")

        test_config = base_config.copy()
        for param, value in config_changes.items():
            test_config[param] = value

        model = ArithmeticTransformer(
            src_vocab_size=test_config["src_vocab_size"],
            tgt_vocab_size=test_config["tgt_vocab_size"],
            d_model=test_config["d_model"],
            num_heads=test_config["num_heads"],
            d_ff=test_config["d_ff"],
            num_layers=test_config["num_layers"],
            dropout=test_config["dropout"],
            max_len=test_config["max_len"],
        )
        model = model.to(DEVICE)

        test_config["epochs"] = min(test_config["epochs"], 5)
        history = train_model(model, dataloaders["train"], dataloaders["val"], test_config)

        test_metrics = evaluate_model(model, dataloaders["test"], nn.CrossEntropyLoss(ignore_index=0))
        gen_metrics = evaluate_model(model, dataloaders["generalization"], nn.CrossEntropyLoss(ignore_index=0))

        ablation_results[config_name] = {
            "config": test_config,
            "val_exact_match": history["val_exact_match"][-1],
            "val_digit_accuracy": history["val_digit_accuracy"][-1],
            "test_exact_match": test_metrics["exact_match"],
            "test_digit_accuracy": test_metrics["digit_accuracy"],
            "gen_exact_match": gen_metrics["exact_match"],
            "gen_digit_accuracy": gen_metrics["digit_accuracy"],
        }

        print(f"\nAblation results for {config_name}:")
        print(f"  Validation Exact Match: {ablation_results[config_name]['val_exact_match']:.4f}")
        print(f"  Test Exact Match: {ablation_results[config_name]['test_exact_match']:.4f}")
        print(f"  Generalization Exact Match: {ablation_results[config_name]['gen_exact_match']:.4f}")

    return ablation_results


def visualize_ablation_metrics(ablation_results):
    sns.set_theme()
    config_names = list(ablation_results.keys())
    val_scores = [r["val_exact_match"] for r in ablation_results.values()]
    test_scores = [r["test_exact_match"] for r in ablation_results.values()]
    gen_scores = [r["gen_exact_match"] for r in ablation_results.values()]

    x = np.arange(len(config_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, val_scores, width, label="Validation")
    rects2 = ax.bar(x, test_scores, width, label="Test")
    rects3 = ax.bar(x + width, gen_scores, width, label="Generalization")

    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig("ablation_results.png")
    print("Saved ablation_results.png")
    plt.close()

    print("\nDetailed Ablation Study Results:")
    print("-" * 100)
    print(f"{'Configuration':<20} | {'Val Exact Match':<15} | {'Test Exact Match':<15} | {'Gen Exact Match':<15}")
    print("-" * 100)
    for config_name, results in ablation_results.items():
        print(f"{config_name:<20} | {results['val_exact_match']:<15.4f} | {results['test_exact_match']:<15.4f} | {results['gen_exact_match']:<15.4f}")


def analyze_ablation_performance(ablation_results):
    baseline = ablation_results.get("baseline", None)
    if not baseline:
        print("Baseline results not found in ablation study.")
        return

    baseline_val = baseline["val_exact_match"]
    baseline_test = baseline["test_exact_match"]
    baseline_gen = baseline["gen_exact_match"]

    results_table = []
    for config_name, results in ablation_results.items():
        if config_name == "baseline":
            continue
        val_change = (results["val_exact_match"] - baseline_val) / max(baseline_val, 1e-8) * 100
        test_change = (results["test_exact_match"] - baseline_test) / max(baseline_test, 1e-8) * 100
        gen_change = (results["gen_exact_match"] - baseline_gen) / max(baseline_gen, 1e-8) * 100
        avg_change = (val_change + test_change + gen_change) / 3

        param_changes = []
        for param, value in results["config"].items():
            if param in baseline["config"] and value != baseline["config"][param]:
                param_changes.append(f"{param}: {baseline['config'][param]} → {value}")

        results_table.append({
            "config_name": config_name,
            "param_changes": ", ".join(param_changes),
            "val_change": val_change,
            "test_change": test_change,
            "gen_change": gen_change,
            "avg_change": avg_change,
        })

    results_table.sort(key=lambda x: x["avg_change"], reverse=True)

    print("\nImpact of Model Changes (% change from baseline):")
    print("-" * 100)
    print(f"{'Configuration':<15} | {'Parameter Changes':<30} | {'Val':<8} | {'Test':<8} | {'Gen':<8} | {'Avg':<8}")
    print("-" * 100)
    for result in results_table:
        print(f"{result['config_name']:<15} | {result['param_changes']:<30} | {result['val_change']:+8.2f}% | {result['test_change']:+8.2f}% | {result['gen_change']:+8.2f}% | {result['avg_change']:+8.2f}%")

    best_val = max(ablation_results.items(), key=lambda x: x[1]["val_exact_match"])
    best_test = max(ablation_results.items(), key=lambda x: x[1]["test_exact_match"])
    best_gen = max(ablation_results.items(), key=lambda x: x[1]["gen_exact_match"])

    print(f"\nRecommendations based on ablation studies:")
    print(f"  Best validation performance: {best_val[0]} ({best_val[1]['val_exact_match']:.4f})")
    print(f"  Best test performance: {best_test[0]} ({best_test[1]['test_exact_match']:.4f})")
    print(f"  Best generalization performance: {best_gen[0]} ({best_gen[1]['gen_exact_match']:.4f})")

    positive_configs = [r for r in results_table if r["avg_change"] > 0]
    if positive_configs:
        print("\nModel improvements to consider:")
        for config in positive_configs[:3]:
            print(f"  - {config['config_name']}: {config['param_changes']} (avg. improvement: {config['avg_change']:.2f}%)")

# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def run_training_pipeline(model, dataset_dict, config, idx_to_char):
    dataloaders = prepare_dataloaders(dataset_dict, config["batch_size"])

    print("Starting model training…")
    model = model.to(DEVICE)

    history = train_model(model, dataloaders["train"], dataloaders["val"], config)
    plot_training_metrics(history)

    best_model_path = "checkpoints/best_model.pth"
    if os.path.exists(best_model_path):
        checkpoint = load_checkpoint(best_model_path, model)
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with val accuracy: {checkpoint['metrics']['exact_match']:.4f}")

    test_results = test_model(model, dataloaders["test"], dataloaders["generalization"])

    print("\nAnalyzing model errors…")
    analyze_errors(model, dataloaders["test"], idx_to_char)

    return model, history, test_results


def execute_training_pipeline():
    generate_datasets_if_needed()

    train_list, test_list, val_list, gen_list = load_math_datasets()

    X_train, y_train, char_to_idx, idx_to_char = preprocess_data(train_list)
    X_val, y_val, _, _ = preprocess_data(val_list)
    X_test, y_test, _, _ = preprocess_data(test_list)
    X_gen, y_gen, _, _ = preprocess_data(gen_list)

    with open("vocab.json", "w") as f:
        json.dump(char_to_idx, f)
    with open("rev_vocab.json", "w") as f:
        json.dump(idx_to_char, f)

    config = {
        "src_vocab_size": 15,
        "tgt_vocab_size": 15,
        "d_model": 128,
        "num_heads": 8,
        "d_ff": 512,
        "num_layers": 3,
        "dropout": 0.1,
        "max_len": 20,
        "batch_size": 128,
        "learning_rate": 0.0005,
        "epochs": 10,
    }

    dataset_dict = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_gen": X_gen, "y_gen": y_gen,
    }

    ablation_configs = {
        "baseline": {},
        "smaller_model": {"d_model": 64, "d_ff": 256},
        "larger_model": {"d_model": 256, "d_ff": 1024},
        "fewer_layers": {"num_layers": 2},
        "more_layers": {"num_layers": 4},
        "fewer_heads": {"num_heads": 4},
        "more_heads": {"num_heads": 16},
        "higher_dropout": {"dropout": 0.3},
        "no_dropout": {"dropout": 0.0},
    }

    print("\n=== Training Baseline Model ===")
    baseline_model = ArithmeticTransformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    )

    baseline_model, history, test_results = run_training_pipeline(baseline_model, dataset_dict, config, idx_to_char)

    print("\n=== Running Ablation Studies ===")
    ablation_results = execute_ablation_studies(dataset_dict, config, ablation_configs)

    visualize_ablation_metrics(ablation_results)

    print("\n=== Detailed Analysis of Ablation Results ===")
    analyze_ablation_performance(ablation_results)

    return baseline_model, history, test_results, ablation_results


if __name__ == "__main__":
    model, history, test_results, ablation_results = execute_training_pipeline()
