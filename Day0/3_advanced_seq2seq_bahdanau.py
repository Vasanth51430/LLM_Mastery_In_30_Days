import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import spacy
import random
import math
import time
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
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

# Modify vocabulary creation
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

# Model architecture
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        predicted = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        
        return predicted, hidden, cell

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs

# Model hyperparameters
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TRG_VOCAB)
ENCODER_EMBED_DIM = 256
DECODER_EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Initialize the model
encoder = Encoder(INPUT_DIM, ENCODER_EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, ENC_DROPOUT)
attention = Attention(HIDDEN_DIM)
decoder = Decoder(OUTPUT_DIM, DECODER_EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DEC_DROPOUT, attention)
model = Seq2SeqAttention(encoder, decoder)

logging.info(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Define optimizer and loss
optimizer = optim.Adam(model.parameters())
PAD_IDX = SRC_VOCAB[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for sample in batch:
        src_batch.append(torch.tensor([SRC_VOCAB.get(token, SRC_VOCAB[UNK_TOKEN]) for token in sample['src']]))
        trg_batch.append(torch.tensor([TRG_VOCAB.get(token, TRG_VOCAB[UNK_TOKEN]) for token in sample['trg']]))
    
    src_batch = pad_sequence(src_batch, padding_value=SRC_VOCAB[PAD_TOKEN])
    trg_batch = pad_sequence(trg_batch, padding_value=TRG_VOCAB[PAD_TOKEN])
    
    return src_batch, trg_batch

# Training function
def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Evaluating", leave=False):
            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Function to generate a translation
def translate_sentence(sentence, model, max_len=50):
    model.eval()
    
    tokens = tokenize_de(sentence)
    tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
    src_indexes = [SRC_VOCAB.get(token, SRC_VOCAB[UNK_TOKEN]) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [TRG_VOCAB[SOS_TOKEN]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).unsqueeze(0)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == TRG_VOCAB[EOS_TOKEN]:
            break

    trg_tokens = [list(TRG_VOCAB.keys())[list(TRG_VOCAB.values()).index(i)] for i in trg_indexes]

    return trg_tokens[1:-1]  # Exclude <sos> and <eos> tokens

# Create DataLoaders
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Training loop
N_EPOCHS = 10
CLIP = 1

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
#     start_time = time.time()
    
#     train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, val_dataloader, criterion)

#     end_time = time.time()
    
#     epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'best_model.pt')
    
#     logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pt'))

# Example translations
for example_idx in range(3):  # Change the range to translate more examples
    src = train_data[example_idx]['src']
    trg = train_data[example_idx]['trg']

    print(f'src = {" ".join(src)}')
    print(f'trg = {" ".join(trg)}')

    translation = translate_sentence(" ".join(src), model)
    print(f'predicted trg = {" ".join(translation)}')
    print()