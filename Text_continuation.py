# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Training data
text = """
the cat sat on the mat.
the dog ran through the park.
the bird flew over the trees.
the cat watched the rain fall.
the dog chased the ball.
the sun shone in the sky.
the children played in the garden.
the cat slept on the couch.
the dog barked at the mailman.
the bird sang a song.
"""

# Preprocess text
def prepare_data():
    words = text.lower().replace(".", "").split()
    vocab = list(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    data = [word2idx[w] for w in words]

    seq_len = 2

    X = []
    Y= []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])

    X = torch.tensor(X)
    Y = torch.tensor(Y)

# Neural network
def create_model(vocab_size):
    emb = nn.Embedding(vocab_size, 8)
    lstm = nn.LSTM(8, 16, batch_first=True)
    fc = nn.Linear(16, vocab_size)
    return emb, lstm, fc

# Training function
def train_model(X, Y, emb, lstm, fc, epochs=300):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        list(emb.parameters()) +
        list(lstm.parameters()) +
        list(fc.parameters()),
        lr=0.01
    )

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = forward(X, emb, lstm, fc)
        loss = criterion(outputs, Y)

        loss.backward()
        optimizer.step()

# Sentence completion

