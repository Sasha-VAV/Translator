import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, tqdm

from Translator import Writer


def mlflow_decorator(func: callable) -> callable:
    def wrapper(*args):
        mlflow.set_tracking_uri(uri="http://localhost:5000")
        with mlflow.start_run():
            func(*args)

    return wrapper


@mlflow_decorator
def train_decoder():
    dataset = load_dataset("roneneldan/TinyStories")["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = Writer.from_pretrained("Sashavav/Translator").to(device)
    model = writer.model
    tokenizer = writer.tokenizer
    # model.load_state_dict(torch.load("../data/text_generator.pth"))
    max_sequence_length = 128
    # Train args
    epochs = 10
    scaler = torch.amp.GradScaler(device=device.type)
    num_of_sequences = 2119719
    save_every_n_steps = 100
    effective_batch_size = 1200
    batch_size = 400
    acc_steps = effective_batch_size // batch_size
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    batch_i = 0
    acc_i = 0
    save_i = 0
    seq_i = 0
    batch = torch.zeros(
        batch_size, max_sequence_length, dtype=torch.long, device=device
    )
    optimizer.zero_grad()
    running_loss = 0.0
    for epoch in range(epochs):
        for story in tqdm(dataset, total=num_of_sequences):
            if batch_i < batch_size:
                batch[batch_i] = tokenizer.tokenize(story["text"], device=device)
                batch_i += 1
                continue
            x_batch = batch[:, :-1]
            y_batch = batch[:, 1:]
            with torch.amp.autocast(device_type=device.type):
                y_pred = model(x_batch)
                y_batch = y_batch.reshape(-1)
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                loss = criterion(y_pred, y_batch)
            # loss.backward()
            running_loss += loss.item()
            scaler.scale(loss).backward()
            acc_i += 1
            if acc_i == acc_steps:
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()
                acc_i = 0
                mlflow.log_metric("loss", running_loss / acc_steps)
                running_loss = 0.0
            save_i += 1
            if save_i == save_every_n_steps:
                writer.save_pretrained("train")
                save_i = 0
            seq_i += 2
            if seq_i >= num_of_sequences:
                break
            batch_i = 0
        writer.push_to_hub()
        writer.save_pretrained(f"epoch{epoch + 1}")
    mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    train_decoder()
