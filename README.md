# Translator
This is a research project to create a translator from an article Attention Is All You Need. 
At current state, I don't have enough resources to train a model for this task, 
so I'm presenting only the decoder, that can generate some text based on the input.

# How to launch
- Clone repository
- Run code
```python
from Translator import Writer
writer = Writer.from_pretrained("Sashavav/Translator") #  .to("cuda")
print(writer(input_seq="One day I saw a "))
```