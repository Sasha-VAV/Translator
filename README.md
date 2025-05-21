# Translator
This is a research project to create a model that can work with text

### How to launch
- Clone repository
- Run code
```python
from Translator import Writer
writer = Writer.from_pretrained() #  .to("cuda")
print(writer(input_seq="One day I saw a ", temperature=2))  # I highly recommend high temperature
```

# Model architecture and training pipeline
Transformer decoder architecture with params:
- decoder blocks = 4
- vocab size = 8192
- embedding_size = 512
- number of heads = 8
- hidden size in FFN = 1024
- max_sequence_length = 128

Trained with params:
- loss = CrossEntropyLoss
- optimizer = Adam
- batch = 400
- accumulation steps = 3
- epochs = 10
- nums of sequences in dataset = 21kk

Total training time: 10 hours

# Sources
- Architecture inspired from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)