import random

def generate_sequences(num_sequences, max_length, N):
    data = []
    for _ in range(num_sequences):
        seq_len = random.randint(2, max_length)
        sequence = [random.randint(0, N) for _ in range(seq_len)]
        data.append(sequence)
    return data

N = 100  # integers from 0 to 100
sequences = generate_sequences(10000, 10, N)

def preprocess_data(sequences):
    X = []
    Y = []
    for seq in sequences:
        X.append(seq[:-1])
        Y.append(seq[-1])
    return X, Y

X, Y = preprocess_data(sequences)

import torch
import torch.nn as nn
import torch.optim as optim

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = N + 1  # using one-hot encoding for input
hidden_size = 128
output_size = N + 1
num_layers = 2

model = RNNModel(input_size, hidden_size, output_size, num_layers)

# Hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert sequences to one-hot encoding
def one_hot_encode(sequences, N, max_length=None):
    if not max_length:
        max_length = max([len(seq) for seq in sequences])
    encoded = []
    for seq in sequences:
        encoding = torch.zeros(max_length, N+1)
        for i, num in enumerate(seq):
            encoding[i][num] = 1
        encoded.append(encoding)
    return torch.stack(encoded)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_sequences = X_train[i:i+batch_size]
        max_length_in_batch = max([len(seq) for seq in batch_sequences])
        
        inputs = one_hot_encode(batch_sequences, N, max_length_in_batch)
        labels = torch.tensor(Y_train[i:i+batch_size])
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test on test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        batch_sequences = X_test[i:i+batch_size]
        max_length_in_batch = max([len(seq) for seq in batch_sequences])
        
        inputs = one_hot_encode(batch_sequences, N, max_length_in_batch)
        labels = torch.tensor(Y_test[i:i+batch_size])
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test data: {100 * correct / total}%')


