import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn


class Embedding(nn.Module):
    """
    Embedding block to convert input vectors into their embeddings
    """

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        max_len: int = 512,
        padding_idx: int = 0,
        is_cls: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the embedding block,
        :param num_embeddings: vocabulary size,
        :param embed_dim: dimension of embeddings,
        :param max_len: maximum length of a sequence,
        :param padding_idx: index of a padding token,
        :param is_cls: should cls token be added in the beginning?
        """
        super().__init__(*args, **kwargs)
        # Embedding look-up table
        self.embedding = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=padding_idx
        )
        # Adding learnable cls token
        if is_cls:
            self.cls = nn.Parameter(torch.rand(1, 1, embed_dim))
            max_len += 1
        # Getting positional embeddings
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        divider = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * divider)
        pe[:, 1::2] = torch.cos(pos * divider)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.is_cls = is_cls
        self.padding_idx = padding_idx

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prepare masks
        pad_mask = x == self.padding_idx
        if self.is_cls:
            cls = torch.zeros(x.size(0), 1, device=x.device).bool()
            pad_mask = torch.cat([cls, pad_mask], dim=1)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(
                pad_mask.size(-1), pad_mask.size(-1), device=x.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        causal_mask = pad_mask | causal_mask
        pad_mask = pad_mask.to(x.device)
        causal_mask = causal_mask.to(x.device)
        # Embed seq
        x = self.embedding(x)
        if self.is_cls:
            cls = self.cls.repeat(x.size(0), 1, 1)
            x = torch.cat((cls, x), dim=1)
        x += self.pe[:, : x.size(1) + self.is_cls]
        return x, pad_mask, causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention block, that computes self-attention
    """

    def __init__(self, embed_dim: int, num_heads: int, *args, **kwargs):
        """
        Initialize the multi-head attention block,
        :param embed_dim: dimension of embeddings,
        :param num_heads: the number of heads
        """
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        if self.embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes multi-head attention,
        :param query: tensor BxTxD that represents query matrix in mha
        :param key: tensor BxTxD that represents key matrix in mha
        :param value: tensor BxTxD that represents value matrix in mha
        :param mask: tensor BxTxT that will be applied before softmax
        :return: tensor BxTxD that represents computed attention
        """
        # Split into heads
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads)
        # attention scores
        x = torch.matmul(query, key.transpose(-2, -1))
        x /= math.sqrt(self.head_dim)
        # filling -inf to mask values before softmax
        if mask is not None:
            x = x.masked_fill(mask, float("-inf"))
        x = self.softmax(x)
        # multiplying attention scores with values
        x = torch.matmul(x, value)
        # concatenating heads
        x = rearrange(x, "b h n d -> b n (h d)", h=self.num_heads)
        # a linear block to scale
        x = self.linear(x)
        return x


class FeedForward(nn.Module):
    """
    MLP for transformer block
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        act_func: callable = nn.GELU,
        *args,
        **kwargs,
    ):
        """
        Initializes the feedforward block,
        :param embed_dim: number of input and output neurons,
        :param hidden_size: number of hidden neurons
        """
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_dim)
        self.act_func = act_func()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feedforward block
        :param x: tensor BxTxD that represents input seq
        :return: tensor BxTxD that represents output seq
        """
        x = self.fc1(x)
        x = self.act_func(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """
    Represents a transformer block
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        is_cross_attention: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the transformer block,
        :param embed_dim: dimension of embeddings,
        :param num_heads: number of heads in attention
        :param hidden_size: size of hidden layer in feed forward block
        :param dropout: probability of neuron to be dropped
        :param is_cross_attention: is this block a cross attention block
        """
        # Initializing params, checkout forward for their description
        super().__init__(*args, **kwargs)
        self.is_cross_attention = is_cross_attention
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_size)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.is_cross_attention:
            self.norm3 = nn.LayerNorm(embed_dim)
            self.dropout3 = nn.Dropout(dropout)
            self.cross_q = nn.Linear(embed_dim, embed_dim)
            self.cross_k = nn.Linear(embed_dim, embed_dim)
            self.cross_v = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.self_q = nn.Linear(embed_dim, embed_dim)
        self.self_k = nn.Linear(embed_dim, embed_dim)
        self.self_v = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        memory: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block
        :param x: tensor BxTxD that represents input seq
        :param pad_mask: tensor BxTxT that represents the padding mask
        :param causal_mask: tensor BxTxT that represents the causal mask combined with padding mask
        :param memory: tensor BxTxD that represents input seq in the cross-attention block
        :return: tensor BxTxD that represents output seq
        """
        if self.is_cross_attention and memory is None:
            raise ValueError("memory must be provided for cross_attention")
        # pre-norm
        tmp = self.norm1(x)
        # q,k,v matrices
        q = self.self_q(tmp)
        k = self.self_k(tmp)
        v = self.self_v(tmp)
        # computing multi-head attention
        self_attn = self.mha(q, k, v, causal_mask)
        # adding with dropout
        x = x + self.dropout1(self_attn)
        # for cross-attention block applying the same pipeline,
        # but q,k,v are computed differently and k,v came from the memory
        if self.is_cross_attention:
            tmp = self.norm2(x)
            q = self.cross_q(tmp)
            k = self.cross_k(memory)
            v = self.cross_v(memory)
            cross_attn = self.mha(q, k, v, pad_mask)
            x = x + self.dropout2(cross_attn)
            # applying pre-norm ffn with dropout
            fnn = self.ffn(self.norm3(x))
            x = x + self.dropout3(fnn)
        else:
            # applying pre-norm ffn with dropout
            fnn = self.ffn(self.norm2(x))
            x = x + self.dropout2(fnn)
        return x


class Encoder(nn.Module):
    """
    Encoder transformer
    """

    def __init__(
        self,
        n_blocks: int,
        num_embeddings: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        max_len: int = 512,
        pad_idx: int = 0,
        num_classes: Optional[int] = None,
        is_cls: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the encoder block,
        :param n_blocks: number of sequential blocks,
        :param num_embeddings: vocabulary size,
        :param embed_dim: dimension of embeddings,
        :param num_heads: number of heads in multi-head attention,
        :param hidden_size: size of hidden layer in feed forward block,
        :param max_len: maximum sequence length,
        :param pad_idx: index of a padding token,
        :param num_classes: number of classes on the output,
        :param is_cls: is cls token exists and participate in classification task,
        WARNING: cls token is expensive to train
        """
        super().__init__(*args, **kwargs)
        self.n_blocks = n_blocks
        self.embedding = Embedding(
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            max_len=max_len,
            padding_idx=pad_idx,
            is_cls=is_cls,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    *args,
                    **kwargs,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.is_cls = is_cls
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get embeddings and pad_mask
        x, pad_mask, _ = self.embedding(x)
        # compute new embeddings sequentially
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask, causal_mask=pad_mask)
        # if it is a classification task
        if self.num_classes is not None and self.is_cls:
            x = x[:, 0]
        elif self.num_classes is not None and not self.is_cls:
            x = x.mean(dim=1)
            x = self.linear(x)
        return x


class Decoder(nn.Module):
    """
    Decoder transformer
    """

    def __init__(
        self,
        n_blocks: int,
        num_embeddings: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        max_len: int = 512,
        pad_idx: int = 0,
        is_cross_attention: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the decoder block,
        :param n_blocks: number of sequential blocks,
        :param num_embeddings: vocabulary size,
        :param embed_dim: dimension of embeddings,
        :param num_heads: number of heads in multi-head attention,
        :param hidden_size: size of hidden layer in feed forward block,
        :param max_len: maximum sequence length,
        :param pad_idx: index of a padding token,
        :param is_cross_attention: should this block be a cross attention block?
        WARNING: is_cross_attention requires to be used in EncoderDecoder, not in a pure Decoder
        """
        super().__init__(*args, **kwargs)
        self.n_blocks = n_blocks
        self.max_seq_len = max_len
        self.embedding = Embedding(
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            max_len=max_len,
            padding_idx=pad_idx,
            is_cls=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    hidden_size,
                    is_cross_attention,
                    *args,
                    **kwargs,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.linear = nn.Linear(embed_dim, num_embeddings)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, memory: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the decoder
        :param x: tensor BxTxD that represents input seq
        :param memory: tensor BxTxD that represents input seq in the cross-attention block
        :return: tensor BxTxD that represents output seq
        """
        x, pad_mask, causal_mask = self.embedding(x)
        for block in self.blocks:
            x = block(
                x,
                pad_mask=pad_mask,
                causal_mask=causal_mask,
                memory=memory,
            )
        x = self.linear(x)
        return x

    def predict(
        self, x: torch.Tensor, temperature: float = 1.0, memory: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict next token with given temperature,
        :param x: tensor BxTxD that represents input seq
        :param temperature: float, temperature parameter
        :param memory: tensor BxTxD that represents input seq in the cross-attention block
        :return: tensor Bx1 with next tokens
        """
        x = self.forward(
            x.unsqueeze(0), memory=memory.unsqueeze(0) if memory else None
        ).squeeze(0)[-1]
        x = self.softmax(x / temperature)
        x = torch.multinomial(x, 1)
        return x

    def generate_sequence(
        self, x: torch.Tensor, max_size: int = 512, temperature: float = 0.7
    ) -> torch.Tensor:
        """
        Method do generate a sequence, until <eos> token has been found or reached limit
        :param x: tensor TxD that represents input seq,
        :param max_size: int, maximum sequence length,
        :param temperature: float, temperature parameter
        WARNING: it does not support batches
        """
        if max_size > self.max_seq_len:
            raise ValueError(
                f"max_seq_len must be less than or equal to {self.max_seq_len}"
            )
        k = 0
        # searching for the first eos or pad token
        for tmp in x:
            if tmp == 3:
                x[k] = 0
                break
            if tmp == 0:
                break
            k += 1
        # starting generating from it
        for i in range(k, max_size):
            next_token = self.predict(x, temperature)
            x[i] = next_token
            if next_token == 3:
                break
        return x


class EncoderDecoder(nn.Module):
    """
    Wrapper class to work with an encoder and decoder
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, *args, **kwargs):
        """
        Initialize with prepared an encoder and decoder transformers,
        :param encoder: should not be set up with classification task,
        :param decoder: should have is_cross_attention set to true
        """
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.max_seq_len = self.decoder.max_seq_len

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(x)
        return self.decoder(target, memory=memory)

    def generate_sequence(
        self, x: torch.Tensor, max_size: int = 128, temperature: float = 0.7
    ) -> torch.Tensor:
        """
        Method do generate a sequence, until <eos> token has been found or reached limit
        :param x: tensor TxD that represents input seq,
        :param max_size: int, maximum sequence length,
        :param temperature: float, temperature parameter
        WARNING: it does not support batches
        """
        if max_size > self.max_seq_len:
            raise ValueError(
                f"max_seq_len must be less than or equal to {self.max_seq_len}"
            )
        memory = self.encoder(x.unsqueeze(0)).squeeze(0)
        out = torch.zeros(max_size, device=x.device, dtype=torch.long)
        out[0] = 2
        for i in range(1, max_size):
            next_token = self.decoder.predict(out[:i], temperature, memory=memory)
            out[i] = next_token
            if next_token == 3:
                break
        return out
