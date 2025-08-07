import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
import time
import math
from tqdm import tqdm
import os
import random
import requests
from typing import List

# ==============================================================================
# SECTION 1: CORE MODULE DEFINITIONS
# This section contains the complete, self-contained code for our final
# Encoder-Decoder PLE architecture, configured for batch_first=True operation.
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PLEEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_ple_concepts: int, ple_nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ple_attention = nn.MultiheadAttention(d_model, ple_nhead, dropout=dropout, batch_first=True)
        self.ple_concepts = Parameter(torch.Tensor(num_ple_concepts, d_model))
        self.ple_gate = Parameter(torch.zeros(1))
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.ple_concepts)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        sa_input = self.norm1(src)
        attn_output, _ = self.self_attn(sa_input, sa_input, sa_input, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        ple_input = self.norm2(src)
        ple_kv = self.ple_concepts.unsqueeze(0).repeat(src.shape[0], 1, 1)
        resonated_context, _ = self.ple_attention(query=ple_input, key=ple_kv, value=ple_kv)
        src = src + self.dropout2(resonated_context * self.ple_gate)
        ffn_input = self.norm3(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ffn_input))))
        src = src + self.dropout3(ff_output)
        return src

class PLEDecoderLayerWithCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, ple_nhead: int, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_ple = nn.MultiheadAttention(d_model, ple_nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, ple_memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        sa_input = self.norm1(tgt)
        attn_output, _ = self.self_attn(sa_input, sa_input, sa_input, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        ca_input = self.norm2(tgt)
        cross_attn_output, _ = self.cross_attn_encoder(query=ca_input, key=memory, value=memory, 
                                                       key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        ple_ca_input = self.norm3(tgt)
        ple_cross_attn_output, _ = self.cross_attn_ple(query=ple_ca_input, key=ple_memory, value=ple_memory)
        tgt = tgt + self.dropout3(ple_cross_attn_output)
        ffn_input = self.norm4(tgt)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ffn_input))))
        tgt = tgt + self.dropout4(ff_output)
        return tgt

class EncoderPrimedPLE_GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, 
                 num_ple_concepts: int, ple_nhead: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'EncoderPrimedGenerator'
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            PLEEncoderLayer(d_model, nhead, num_ple_concepts, ple_nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            PLEDecoderLayerWithCrossAttention(d_model, nhead, ple_nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
        encoder_concepts = torch.cat([layer.ple_concepts for layer in self.encoder_layers], dim=0)
        ple_memory = encoder_concepts.unsqueeze(0).repeat(src.shape[0], 1, 1)
        return memory, ple_memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, ple_memory: torch.Tensor, 
               tgt_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, ple_memory, 
                           tgt_mask=tgt_mask, 
                           memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(output)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                tgt_mask: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        memory, ple_memory = self.encode(src, src_key_padding_mask)
        return self.decode(tgt, memory, ple_memory, tgt_mask, src_key_padding_mask)

# ==============================================================================
# SECTION 2: DATA HANDLING FOR CONDITIONED GENERATION TASK
# ==============================================================================

class ConditionedShakespeareDataset(Dataset):
    def __init__(self, text, context_len, target_len):
        self.chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.text_as_int = [self.char_to_int[c] for c in text]
        self.context_len = context_len
        self.target_len = target_len

    def __len__(self):
        return len(self.text_as_int) - (self.context_len + self.target_len)

    def __getitem__(self, idx):
        src = torch.tensor(self.text_as_int[idx : idx + self.context_len], dtype=torch.long)
        tgt = torch.tensor(self.text_as_int[idx + self.context_len : idx + self.context_len + self.target_len], dtype=torch.long)
        return src, tgt

# ==============================================================================
# SECTION 3: EXPERIMENT INFRASTRUCTURE
# ==============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_tgt_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).to(DEVICE)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for src, tgt in pbar:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_mask = create_tgt_mask(tgt_input.size(1))
        src_padding_mask = torch.zeros(src.shape).bool().to(DEVICE)
        logits = model(src, tgt_input, tgt_mask, src_padding_mask)
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

@torch.no_grad()
def generate_conditioned_text(model, dataset, context, max_len=200):
    model.eval()
    context_tensor = torch.tensor([dataset.char_to_int[c] for c in context], dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_padding_mask = torch.zeros(context_tensor.shape).bool().to(DEVICE)
    
    memory, ple_memory = model.encode(context_tensor, src_padding_mask)
    
    # Start with a single newline character as the beginning of the generation
    tgt_input = torch.tensor([[dataset.char_to_int['\n']]], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_len):
        tgt_mask = create_tgt_mask(tgt_input.size(1))
        output_logits = model.decode(tgt_input, memory, ple_memory, tgt_mask, src_padding_mask)
        last_logits = output_logits[:, -1, :]
        next_token_idx = torch.multinomial(F.softmax(last_logits, dim=-1), num_samples=1).item()
        
        next_token_tensor = torch.tensor([[next_token_idx]], dtype=torch.long).to(DEVICE)
        tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

    generated_indices = tgt_input.squeeze(0).tolist()[1:] # Exclude the start token
    return ''.join([dataset.int_to_char.get(i, '?') for i in generated_indices])

# ==============================================================================
# SECTION 4: MAIN EXPERIMENT DRIVER
# ==============================================================================

def main_encoder_primed_generation_experiment():
    print("="*60)
    print("--- Standalone Encoder-Primed Generation Experiment ---")
    print("="*60)

    # --- Data Preparation ---
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text_file = "tinyshakespeare.txt"
    if not os.path.exists(text_file):
        print("Downloading Tiny Shakespeare dataset...")
        r = requests.get(data_url)
        with open(text_file, 'w') as f:
            f.write(r.text)
    
    with open(text_file, 'r') as f:
        text = f.read()
    
    dataset = ConditionedShakespeareDataset(text, context_len=150, target_len=150)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"Dataset created with {dataset.vocab_size} unique characters.")

    # --- Model and Training Configuration ---
    model_config = {
        "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024,
        "num_ple_concepts": 32, "ple_nhead": 4, "dropout": 0.1
    }
    train_config = {
        "epochs": 15, "lr": 0.001
    }

    model = EncoderPrimedPLE_GPT(vocab_size=dataset.vocab_size, **model_config).to(DEVICE)

    print(f"\nModel Initialized on {DEVICE}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    for epoch in range(1, train_config['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{train_config['epochs']} ---")
        train_loss = train_epoch(model, dataloader, optimizer, criterion)
        print(f"  -> Epoch {epoch} | Train loss: {train_loss:.3f}")

    # --- Final Generation ---
    print("\n" + "="*60)
    print("--- Final Conditioned Text Generation Sample ---")
    print("="*60)
    context_prompt = "JULIET:\nO, swear not by the moon, the inconstant moon,\nThat monthly changes in her circled orb,\nLest that thy love prove likewise variable."
    print(f"CONTEXT PROMPT:\n{context_prompt}")
    print("\nMODEL GENERATION:")
    generated_text = generate_conditioned_text(model, dataset, context=context_prompt)
    print(generated_text)

if __name__ == '__main__':
    main_encoder_primed_generation_experiment()
