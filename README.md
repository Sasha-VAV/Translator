# Translator
Model that translates text from English to Russian with Attention Is All You Need transformer

# Data
In this project we used OpenSubtitles English to Russian dataset.

[Link to get from OPUS](https://opus.nlpl.eu/results/en&ru/corpus-result-table)

[Direct download link](https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-ru.txt.zip)

# Tokenizer
We use Sentencepiece, params for the tokenizer:
- Vocabulary length = 10000
- Training pairs = 200000