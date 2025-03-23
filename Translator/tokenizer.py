import sentencepiece as spm
from sentencepiece import SentencePieceProcessor


def create_tokenizer(max_tokens: int = 10000, max_corpus_len: int = 400000):
    with (
        open("../data/OpenSubtitles.en-ru.en", encoding="utf-8") as f_en,
        open("../data/OpenSubtitles.en-ru.ru", encoding="utf-8") as f_ru,
        open("../data/tmp.txt", "w", encoding="utf-8") as f_dump,
    ):
        k = 0

        for en_line, ru_line in zip(f_en, f_ru):
            if k > max_corpus_len:
                break
            else:
                k += 2
            f_dump.writelines([en_line, ru_line])

    spm.SentencePieceTrainer.Train(
        input="../data/tmp.txt",
        model_prefix="../data/spm_model",
        vocab_size=max_tokens,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


class Tokenizer(SentencePieceProcessor):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def Encode(self, input, out_type=None, *args, **kwargs):
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


def load_tokenizer(
    path: str = "../data/spm_model.model", seq_len: int = 512
) -> spm.SentencePieceProcessor:
    sp = Tokenizer(seq_len=seq_len)
    sp.Load(path)
    return sp


if __name__ == "__main__":
    create_tokenizer()
