import pytest

from Translator.tokenizer import load_tokenizer


def test_tokenizer():
    tokenizer = load_tokenizer("../data/spm_model.model")
    test_test = (
        "Hi, my name is alex and I'm grateful to be split into tokens. I can speak russian too:"
        "Привет ребята, как ваши дела? Я чувствую себя восхитительно."
    )
    encoded = tokenizer.Encode(test_test)
    assert tokenizer.Decode(encoded) == test_test
    test_test += "~"
    with pytest.raises(AssertionError):
        assert tokenizer.Decode(tokenizer.Encode(test_test)) == test_test
