import torch
import torch.nn as nn

from Translator.transformer import Block, Embedding, Encoder, MultiHeadAttention


def test_embedding():
    vocab_size = 10000
    embedding_dim = 512
    max_len = 512
    embedding_layer = Embedding(vocab_size, embedding_dim, max_len)
    input_ids = torch.randint(vocab_size, (2, max_len))
    output, _, _ = embedding_layer(input_ids)
    assert output.shape == (*input_ids.shape, embedding_dim)
    embedding_dim = 100
    max_len = 200
    embedding_layer = Embedding(vocab_size, embedding_dim, max_len)
    input_ids = torch.randint(vocab_size, (2, 30))
    output, _, _ = embedding_layer(input_ids)
    assert output.shape == (*input_ids.shape, embedding_dim)


def test_mha():
    embedding_dim = 512
    n_heads = 8
    n_tokens = 8
    batch_size = 32
    my_mha = MultiHeadAttention(embedding_dim, n_heads)
    torch_mha = nn.MultiheadAttention(embedding_dim, n_heads)
    test_tensor = torch.rand(3, batch_size, n_tokens, embedding_dim)
    q, k, v = test_tensor.unbind(dim=0)
    a = my_mha(q, k, v)
    b = torch_mha(q, k, v)[0]
    assert a.shape == b.shape
    criterion = nn.MSELoss()
    loss = criterion(a, a)
    loss.backward()


def test_encoder_block():
    embedding_dim = 512
    n_heads = 8
    n_tokens = 512
    batch_size = 32
    hidden_size = 1024
    encoder_block = Block(embedding_dim, n_heads, hidden_size)
    inputs = torch.rand(batch_size, n_tokens, embedding_dim)
    outputs = encoder_block(inputs, torch.zeros(batch_size, n_tokens), None)
    assert inputs.shape == outputs.shape
    criterion = nn.MSELoss()
    loss = criterion(outputs, outputs)
    loss.backward()


def test_encoder():
    embedding_dim = 128
    n_heads = 4
    n_blocks = 4
    vocab_size = 10000
    n_tokens = 128
    batch_size = 32
    hidden_size = 1024
    n_classes = 2
    encoder = Encoder(
        n_blocks,
        vocab_size,
        embedding_dim,
        n_heads,
        hidden_size,
        max_len=n_tokens,
        num_classes=n_classes,
    )
    inputs = torch.randint(vocab_size, (batch_size, n_tokens))
    outputs = encoder(inputs)
    assert outputs.shape == (batch_size, n_classes)
    criterion = nn.MSELoss()
    loss = criterion(outputs, outputs)
    loss.backward()
