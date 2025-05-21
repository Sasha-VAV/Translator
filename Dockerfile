FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y vim curl git

WORKDIR /app

ENV PYTHONPATH=/app:$PYTHONPATH

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true
RUN poetry install --no-root

COPY . .

CMD ["tail", "-f", "/dev/null"]