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
    
    return X, Y, word2idx, idx2word

# Neural network
def create_model(vocab_size):
    emb = nn.Embedding(vocab_size, 8)
    lstm = nn.LSTM(8, 16, batch_first=True)
    fc = nn.Linear(16, vocab_size)
    return emb, lstm, fc

# Forward pass
def forward(x, emb, lstm, fc):
    x = emb(x)
    output, (hidden, cell) = lstm(x)
    return fc(hidden[-1])

# Training function
def train_model(model, X, Y, epochs=300):
    emb, lstm, fc = model

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
def predict_next(seed, model, word2idx, idx2word, max_len=5):
    emb, lstm, fc = model

    words = seed.lower().split()
    words = [w for w in words if w in word2idx]

    if not words:
        return ""

    for _ in range(max_len):
        context = words[-2:]
        x = torch.tensor([[word2idx[w] for w in context]])

        with torch.no_grad():
            output = forward(x, emb, lstm, fc)
            pred = torch.argmax(output, dim=1).item()
            next_word = idx2word[pred]

        words.append(next_word)

    return " ".join(words)

# Enter sentence chat
def chat(model, word2idx, idx2word):
    while True:
        s = input("Enter a starting phrase: ")
        if s.lower() == "quit":
            break

        sentence = predict_next(s, model, word2idx, idx2word)
        print("Generated:", sentence)
        
if __name__ == "__main__":
    X, Y, w2i, i2w = prepare_data()

    model = create_model(len(w2i))

    train_model(model, X, Y)

    chat(model, w2i, i2w)
    
    
    
    
    
