# Translator
This is a research project to create a model that can work with text

### Use my server
```shell
curl -X POST \
  http://91.211.217.36:4000/write \
  -H "Content-Type: application/json" \
  -d '{"text": "One day I saw a", "temperature": 2}'
```

### How to use in your docker environment
```shell
git clone https://github.com/Sasha-VAV/Translator
docker-compose up -d --build
```
Example request
```shell
curl -X POST \
  http://localhost:4000/write \
  -H "Content-Type: application/json" \
  -d '{"text": "One day I saw a", "temperature": 2}'
```

### How to launch in your environment
- Clone repository
- Install dependencies by
```shell
pip install poetry && poetry install
```
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