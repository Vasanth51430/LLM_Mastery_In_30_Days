import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import gzip
import spacy
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset definition
class Multi30kDataset(Dataset):
    def __init__(self, src_file, trg_file, src_transform=None, trg_transform=None):
        self.src_data = self.load_data(src_file)
        self.trg_data = self.load_data(trg_file)
        self.src_transform = src_transform
        self.trg_transform = trg_transform
        
    def load_data(self, file_path):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = f.readlines()
        return data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_sentence = self.src_data[idx].strip()
        trg_sentence = self.trg_data[idx].strip()
        
        if self.src_transform:
            src_sentence = self.src_transform(src_sentence)
        if self.trg_transform:
            trg_sentence = self.trg_transform(trg_sentence)
        
        return {"src": src_sentence, "trg": trg_sentence}

# Load spaCy models for tokenization
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    logging.error("Spacy models not found. Please download them using:")
    logging.error("python -m spacy download de_core_news_sm")
    logging.error("python -m spacy download en_core_web_sm")
    raise

# Tokenization functions using spaCy
def tokenize_de(text):
    return [token.text.lower() for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]

# Define paths for your datasets
train_de_path = 'res/train.de.gz'
train_en_path = 'res/train.en.gz'
val_de_path = 'res/val.de.gz'
val_en_path = 'res/val.en.gz'
test_de_path = 'res/test_2016_flickr.de.gz'
test_en_path = 'res/test_2016_flickr.en.gz'

# Load datasets
train_data = Multi30kDataset(train_de_path, train_en_path, src_transform=tokenize_de, trg_transform=tokenize_en)
val_data = Multi30kDataset(val_de_path, val_en_path, src_transform=tokenize_de, trg_transform=tokenize_en)
test_data = Multi30kDataset(test_de_path, test_en_path, src_transform=tokenize_de, trg_transform=tokenize_en)

# Define special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# Vocabulary creation
def create_vocab(tokenized_sentences, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Tokenize all sentences
train_de_tokenized = [tokenize_de(sentence.strip()) for sentence in train_data.src_data]
train_en_tokenized = [tokenize_en(sentence.strip()) for sentence in train_data.trg_data]

# Create vocabularies with special tokens
SRC_VOCAB = create_vocab(train_de_tokenized, [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
TRG_VOCAB = create_vocab(train_en_tokenized, [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

logging.info(f"Source vocabulary size: {len(SRC_VOCAB)}")
logging.info(f"Target vocabulary size: {len(TRG_VOCAB)}")

# Transformer components

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, trg_mask):
        attn_output = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        
    def generate_mask(self, src, trg):
        src_mask = (src != SRC_VOCAB[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        trg_mask = (trg != TRG_VOCAB[PAD_TOKEN]).unsqueeze(1).unsqueeze(3)
        seq_length = trg.shape[1]
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        trg_mask = trg_mask & nopeak_mask
        return src_mask, trg_mask
        
    def forward(self, src, trg):
        src_mask, trg_mask = self.generate_mask(src, trg)
        
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * self.scale))
        trg_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(trg) * self.scale))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        dec_output = trg_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, trg_mask)
        
        output = self.fc_out(dec_output)
        return output

# Model hyperparameters
SRC_VOCAB_SIZE = len(SRC_VOCAB)
TRG_VOCAB_SIZE = len(TRG_VOCAB)
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 3
D_FF = 2048
MAX_SEQ_LENGTH = 100
DROPOUT = 0.1

# Initialize the model
model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT)

logging.info(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
PAD_IDX = SRC_VOCAB[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for sample in batch:
        src_batch.append(torch.tensor([SRC_VOCAB.get(token, SRC_VOCAB[UNK_TOKEN]) for token in [SOS_TOKEN] + sample['src'] + [EOS_TOKEN]]))
        trg_batch.append(torch.tensor([TRG_VOCAB.get(token, TRG_VOCAB[UNK_TOKEN]) for token in [SOS_TOKEN] + sample['trg'] + [EOS_TOKEN]]))
    
    src_batch = pad_sequence(src_batch, padding_value=SRC_VOCAB[PAD_TOKEN])
    trg_batch = pad_sequence(trg_batch, padding_value=TRG_VOCAB[PAD_TOKEN])
    
    return src_batch.transpose(0, 1), trg_batch.transpose(0, 1)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    print(len(iterator))
    for src, trg in tqdm(iterator, desc="Training", leave=False):
        optimizer.zero_grad()
        
        output = model(src, trg[:, :-1])
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch
            
            output = model(src, trg[:, :-1])
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    
    tokens = [SOS_TOKEN] + tokenize_de(sentence) + [EOS_TOKEN]
    
    src_indexes = [src_vocab.get(token, src_vocab[UNK_TOKEN]) for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.generate_mask(src_tensor, src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder_embedding(src_tensor)
        for enc_layer in model.encoder_layers:
            enc_src = enc_layer(enc_src, src_mask[0])
    
    trg_indexes = [trg_vocab[SOS_TOKEN]]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        trg_mask = model.generate_mask(src_tensor, trg_tensor)
        
        with torch.no_grad():
            output = model.decoder_embedding(trg_tensor)
            for dec_layer in model.decoder_layers:
                output = dec_layer(output, enc_src, src_mask[0], trg_mask[1])
            output = model.fc_out(output)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)
        
        if pred_token == trg_vocab[EOS_TOKEN]:
            break
    
    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_indexes]
    
    return trg_tokens[1:-1]

# Training loop
N_EPOCHS = 10
CLIP = 1.0
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
#     start_time = time.time()
    
#     train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, val_dataloader, criterion)
    
#     end_time = time.time()
    
#     epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'transformer-translation-model.pt')
    
#     logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Load the best model for evaluation
model.load_state_dict(torch.load('transformer-translation-model.pt'))

# Example translations
for example_idx in range(3):  # Change the range to translate more examples
    src = test_data[example_idx]['src']
    trg = test_data[example_idx]['trg']

    print(f'Source: {" ".join(src)}')
    print(f'Target: {" ".join(trg)}')

    translation = translate_sentence(" ".join(src), SRC_VOCAB, TRG_VOCAB, model, torch.device('cpu' if torch.cuda.is_available() else 'cpu'))
    print(f'Predicted: {" ".join(translation)}')
    print()

# Save the model
torch.save(model.state_dict(), 'final_transformer_translation_model.pt')
print("Model saved as 'final_transformer_translation_model.pt'")