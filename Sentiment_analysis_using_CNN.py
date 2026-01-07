import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextClassificationCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextClassificationCNN, self).__init__()
        # Initialize the embedding layer 
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, 2)
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        # Pass the embedded text through the convolutional layer and apply a ReLU
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2) 
        return self.fc(conved)


# Define the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for sentence, label in data:     
        # Clear the gradients
        model.zero_grad()
        sentence = torch.LongTensor([word_to_ix.get(w, 0) for w in sentence]).unsqueeze(0) 
        label = torch.LongTensor([int(label)])
        outputs = model(sentence)
        loss = criterion(outputs, label)
        loss.backward()
        # Update the parameters
        optimizer.step()
print('Training complete!')



book_reviews = [
    "I love this book".split(),
    "I do not like this book".split()
]
for review in book_reviews:
    # Convert the review words into tensor form
    input_tensor = torch.tensor([word_to_ix[w] for w in review], dtype=torch.long).unsqueeze(0) 
    # Get the model's output
    outputs = model(input_tensor)
    # Find the index of the most likely sentiment category
    _, predicted_label = torch.max(outputs.data, 1)
    # Convert the predicted label into a sentiment string
    sentiment = "Positive" if predicted_label.item()==1 else "Negative"
    print(f"Book Review: {' '.join(review)}")
    print(f"Sentiment: {sentiment}\n")
