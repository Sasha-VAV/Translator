import os
from typing import Optional

import torch
from huggingface_hub import HfApi
from transformers import PretrainedConfig, PreTrainedModel

from .tokenizer import Tokenizer
from .transformer import Decoder, Encoder, EncoderDecoder


def get_reviews_scorer(path_to_tokenizer: str = "data/imdb.model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_blocks = 6
    n_tokens = 50000
    embedding_size = 512
    n_heads = 8
    hidden_size = 2048
    max_sequence_length = 512
    n_classes = 2
    model = Encoder(
        n_blocks=n_blocks,
        num_embeddings=n_tokens,
        embed_dim=embedding_size,
        num_heads=n_heads,
        hidden_size=hidden_size,
        max_len=max_sequence_length,
        num_classes=n_classes,
    ).to(device)
    tokenizer = Tokenizer.load_tokenizer(
        path=path_to_tokenizer, seq_len=max_sequence_length
    )
    return model, tokenizer


def get_text_generator(path_to_tokenizer: str = "data/en-8k.model"):
    n_blocks = 4
    n_tokens = 8192
    embedding_size = 512
    n_heads = 8
    hidden_size = 1024
    max_sequence_length = 128
    model = Decoder(
        n_blocks=n_blocks,
        num_embeddings=n_tokens,
        embed_dim=embedding_size,
        num_heads=n_heads,
        hidden_size=hidden_size,
        max_len=max_sequence_length,
    )
    tokenizer = Tokenizer.load_tokenizer(
        path=path_to_tokenizer, seq_len=max_sequence_length
    )
    return model, tokenizer


def get_translator(
    path_to_input_tokenizer: str = "data/en-10k.model",
    path_to_output_tokenizer: str = "data/ru-10k.model",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_blocks = 6
    n_tokens = 10000
    embedding_size = 512
    n_heads = 8
    hidden_size = 2048
    max_sequence_length = 16
    encoder = Encoder(
        n_blocks=n_blocks,
        num_embeddings=n_tokens,
        embed_dim=embedding_size,
        num_heads=n_heads,
        hidden_size=hidden_size,
        max_len=max_sequence_length,
    ).to(device)
    decoder = Decoder(
        n_blocks=n_blocks,
        num_embeddings=n_tokens,
        embed_dim=embedding_size,
        num_heads=n_heads,
        hidden_size=hidden_size,
        max_len=max_sequence_length,
        is_cross_attention=True,
    )
    model = EncoderDecoder(encoder=encoder, decoder=decoder).to(device)
    input_tokenizer = Tokenizer.load_tokenizer(
        path=path_to_input_tokenizer, seq_len=max_sequence_length
    )
    output_tokenizer = Tokenizer.load_tokenizer(
        path=path_to_output_tokenizer, seq_len=max_sequence_length
    )
    return model, input_tokenizer, output_tokenizer


class WriterConfig(PretrainedConfig):
    model_type = "writer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Writer(PreTrainedModel):
    config_class = WriterConfig
    base_model_prefix = "writer"

    def __init__(self, model, tokenizer):
        super().__init__(WriterConfig())
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device("cpu")

    def save_pretrained(self, save_directory: str, **kwargs):
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        torch.save(
            self.model.state_dict(), os.path.join(save_directory, "pytorch_model.bin")
        )
        self.tokenizer.save_tokenizer(os.path.join(save_directory, "tokenizer.model"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        if not os.path.isdir(pretrained_model_name_or_path):
            from huggingface_hub import snapshot_download

            pretrained_model_name_or_path = snapshot_download(
                repo_id=pretrained_model_name_or_path
            )
            print(pretrained_model_name_or_path)
        model, tokenizer = get_text_generator(
            path_to_tokenizer=os.path.join(
                pretrained_model_name_or_path, "tokenizer.model"
            ),
        )
        model.load_state_dict(
            torch.load(pretrained_model_name_or_path + "/pytorch_model.bin")
        )
        return cls(model, tokenizer)

    def __call__(self, input_seq: str, temperature=0.7):
        tokenized_input = self.tokenizer.tokenize(input_seq, device=self._device)
        outputs = self.model.generate_sequence(
            tokenized_input, temperature=temperature, max_size=128
        )
        return self.tokenizer.Decode(outputs.cpu().tolist())

    def to(self, device: str):
        self.model.to(torch.device(device))
        self._device = torch.device(device)
        return self

    def push_to_hub(self, huggingface_token: Optional[str] = None):
        if huggingface_token is None:
            huggingface_token = os.environ.get("HF_TOKEN")
        self.save_pretrained("huggingface")
        api = HfApi(token=huggingface_token)
        api.upload_folder(
            folder_path="./huggingface",
            repo_id="Sashavav/Translator",
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=os.path.abspath("../README.md"),
            path_in_repo="README.md",
            repo_id="Sashavav/Translator",
            repo_type="model",
        )
