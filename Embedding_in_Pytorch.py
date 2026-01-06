import torch
from torch import nn

# Map a unique index to each word
words = ["This", "book", "was", "fantastic", "I", "really", "love", "science", "fiction", "but", "the", "protagonist", "was", "rude", "sometimes"]
word_to_idx = {word: i for i, word in enumerate(words)}

# Convert word_to_idx to a tensor
inputs = torch.LongTensor([word_to_idx[w] for w in words])

# Initialize embedding layer with ten dimensions
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)

# Pass the tensor to the embedding layer
output = embedding(inputs)
print(output)
