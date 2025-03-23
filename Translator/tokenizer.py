import sentencepiece as spm


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
        input="tmp.txt",
        model_prefix="spm_model",
        vocab_size=max_tokens,
        model_type="bpe",
    )


def load_tokenizer(path: str = "../data/spm_model.model") -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


if __name__ == "__main__":
    create_tokenizer()
