import torch
import torch.nn as nn
import torch.nn.functional as F
# import random

# def generate_sequences(num_sequences, max_length, N):
#     data = []
#     for _ in range(num_sequences):
#         seq_len = random.randint(2, max_length)
#         sequence = [random.randint(0, N) for _ in range(seq_len)]
#         data.append(sequence)
#     return data

# N = 100  # integers from 0 to 100
# sequences = generate_sequences(10000, 10, N)
# print(sequences)
# print("Finished generating sequences")

TOTAL_BASE_PAIRS = 29903 # number of base pairs in the SARS-CoV-2 genome
NUMBER_OF_BASES = 4 # A, C, G, T

vocab_size = NUMBER_OF_BASES * TOTAL_BASE_PAIRS
base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

encode = lambda mutations: [NUMBER_OF_BASES*(mut[0] - 1) + base_map[mut[1]] for mut in mutations]
decode = lambda indices: [((index // NUMBER_OF_BASES),list(base_map.keys())[index % NUMBER_OF_BASES]) for index in indices]

data_file = open("new_paths_2.txt", "r")
data_lines = data_file.readlines()[1:]
data_file.close()
sequences = []
for line in data_lines:
    line = line.strip()
    split_line = line.split("\t")
    if len(split_line) == 2:
        mut_seq = split_line[-1]
        mut_seq_list = mut_seq.split(",")
        mut_seq_tuples = []
        for mut in mut_seq_list:
            substitution = mut[-1]
            position = int(mut[1:-1])
            mut_seq_tuples.append((position, substitution))
        
        sequences.append(encode(mut_seq_tuples))
        

def preprocess_data(sequences):
    X = []
    Y = []
    for seq in sequences:
        X.append(seq[:-1])
        Y.append(seq[-1])
    return X, Y

X, Y = preprocess_data(sequences)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale_factor = d_k ** -0.5

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.attn = ScaledDotProductAttention(self.d_k)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.split_heads(self.W_Q(Q)), self.split_heads(self.W_K(K)), self.split_heads(self.W_V(V))
        
        attn_output, attn_weights = self.attn(Q, K, V, mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(Q.size(0), -1, self.num_heads * self.d_k)
        
        return self.fc(attn_output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def positional_encoding(seq_len, d_model, device):
    position = torch.arange(seq_len).unsqueeze(1).float().to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).to(device)
    pos_enc = position * div_term
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=1)
    return pos_enc.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = positional_encoding(max_seq_len, d_model, device)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_enc[:, :x.size(1), :]
        for block in self.blocks:
            x = block(x)
        return self.fc(x)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_len = 500  # adjust based on maximum sequence length in your dataset
dropout_prob = 0.1

model = GPT2(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout_prob).to(device)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 64

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        batch_sequences = X_train[i:i+batch_size]
        
        inputs = [torch.tensor(seq[:-1], dtype=torch.long) for seq in batch_sequences]  # excluding last element for input
        targets = [torch.tensor(seq[1:], dtype=torch.long) for seq in batch_sequences] # excluding first element for target

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(device)
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape outputs and targets for loss computation
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(X_train)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

# Testing Loop
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_sequences = X_test[i:i+batch_size]

        inputs = [torch.tensor(seq[:-1], dtype=torch.long) for seq in batch_sequences]
        labels = [seq[-1] for seq in batch_sequences]  # last element as label

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(device)
        outputs = model(inputs)
        
        _, predicted = outputs[:, -1, :].max(1)
        
        labels = torch.tensor(labels).to(device)
        
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f'Accuracy on test data: {accuracy:.2f}%')
