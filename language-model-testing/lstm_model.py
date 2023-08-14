import random

def generate_sequences(num_sequences, max_length, N):
    data = []
    for _ in range(num_sequences):
        seq_len = random.randint(2, max_length)
        sequence = [random.randint(0, N) for _ in range(seq_len)]
        data.append(sequence)
    return data

N = 10000  # integers from 0 to 10000
sequences = generate_sequences(1000000, 10, N)
print("Finished generating sequences")

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


class LSTMModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Pass the input through the embedding layer
        x = self.embedding(x)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)
        
        # Pass through a dropout layer
        out = self.dropout(out)
        
        # Pass through the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out

embedding_dim = 50  # dimension of the embedding vectors
hidden_size = 256  # number of features in the LSTM hidden state
num_layers = 3  # number of stacked LSTM layers
dropout_prob = 0.3  # dropout probability

model = LSTMModel(N+1, embedding_dim, hidden_size, N+1, num_layers, dropout_prob)

# Hyperparameters
batch_size = 256
num_epochs = 50  # increased epochs for better accuracy
learning_rate = 0.005
clip = 5  # gradient clipping

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print("Finished splitting data")

print("Training model")

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_sequences = X_train[i:i+batch_size]
        max_length_in_batch = max([len(seq) for seq in batch_sequences])
        
        inputs = [torch.tensor(seq) for seq in batch_sequences]
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(Y_train[i:i+batch_size])
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
    avg_loss = total_loss / len(X_train)
    scheduler.step(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


# Test on test set
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_sequences = X_test[i:i+batch_size]
        
        # Since sequences can have different lengths, we pad them to the max length within each batch
        inputs = [torch.tensor(seq) for seq in batch_sequences]
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        
        labels = torch.tensor(Y_test[i:i+batch_size])
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f'Accuracy on test data: {accuracy:.2f}%')
