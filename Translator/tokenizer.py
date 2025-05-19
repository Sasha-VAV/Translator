import sentencepiece as spm
import torch
from sentencepiece import SentencePieceProcessor


def create_tokenizer(
    max_tokens: int = 10000,
    max_corpus_len: int = 800000,
    tokenizer_name: str = "spm_model",
):
    """
    Legacy method to create a tokenizer from a corpus
    :param max_tokens: vocab len
    :param max_corpus_len: max rows in a corpus to process, leave -1 to process every row
    :param tokenizer_name: name of tokenizer to save
    """
    with (
        open("../data/OpenSubtitles.en-ru.en", encoding="utf-8") as f_en,
        open("../data/OpenSubtitles.en-ru.ru", encoding="utf-8") as f_ru,
        open("../data/tmp.txt", "w", encoding="utf-8") as f_dump,
    ):
        k = 0

        for en_line, ru_line in zip(f_en, f_ru):
            if k > max_corpus_len > 0:
                break
            else:
                k += 1
            f_dump.writelines([ru_line])  # , ru_line])

    spm.SentencePieceTrainer.Train(
        input="../data/tmp.txt",
        model_prefix=f"../data/{tokenizer_name}",
        vocab_size=max_tokens,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


class Tokenizer(SentencePieceProcessor):
    """
    Tokenizer class to process data
    """

    def __init__(self, seq_len):
        """
        Initialize the tokenizer class
        :param seq_len: length in tokens that input should have
        """
        super().__init__()
        self.seq_len = seq_len

    def Encode(self, input, out_type=None, *args, **kwargs):
        """
        Wrapper method, not for direct use, checkout tokenize method
        """
        if out_type is None:
            out_type = self._out_type
        encoded_seq = super().Encode(
            input, out_type=out_type, add_bos=True, add_eos=True, *args, **kwargs
        )
        if len(encoded_seq) > self.seq_len:
            return encoded_seq[: self.seq_len]
        if out_type == int:
            encoded_seq.extend([0] * (self.seq_len - len(encoded_seq)))
        elif out_type == str:
            encoded_seq.extend(["<pad>"] * (self.seq_len - len(encoded_seq)))
        return encoded_seq

    def tokenize(
        self, input: str, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Method to tokenize a sequence into torch tensor
        :param input: sequence to be tokenized
        :param device: torch device, either cpu or cuda
        :return: torch tensor containing tokens
        """
        return torch.tensor(self.Encode(input), dtype=torch.long, device=device)

    @classmethod
    def load_tokenizer(
        cls, path: str = "../data/spm_model.model", seq_len: int = 512
    ) -> "Tokenizer":
        """
        Class method to load a tokenizer from a file
        :param path: path to the tokenizer file, should have ".model" extension
        :param seq_len: length in tokens that tokenized text should have
        :return: SentencePieceProcessor instance
        """
        sp = cls(seq_len=seq_len)
        sp.Load(path)
        return sp


if __name__ == "__main__":
    create_tokenizer(max_tokens=10000, tokenizer_name="ru-10k")
