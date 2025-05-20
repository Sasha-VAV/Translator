import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from Translator import get_reviews_scorer, get_text_generator, get_translator


def train_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    with mlflow.start_run():
        # Dataset path
        df = pd.read_csv("../data/IMDB Dataset.csv")[:50000]

        model, tokenizer = get_reviews_scorer(path_to_tokenizer="../data/imdb.model")
        # model.load_state_dict(torch.load("../data/imdb_scorer.pth"))
        max_sequence_length = 512
        # Train args
        epochs = 5
        effective_batch_size = 100
        batch_size = 32
        acc_steps = effective_batch_size // batch_size
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        x = df.review
        y = (df.sentiment == "positive").astype(int)
        df_nums = x.shape[0] // batch_size
        tmp_x = x[: df_nums * batch_size]
        x = torch.zeros(
            (df_nums * batch_size, max_sequence_length), dtype=torch.long, device=device
        )
        for i, tmp in tqdm(enumerate(tmp_x)):
            x[i] = torch.tensor(tokenizer.Encode(tmp), dtype=torch.long).to(device)
        y = torch.tensor(y[: df_nums * batch_size], dtype=torch.long).to(device)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True, random_state=42
        )
        train_nums = x_train.shape[0]
        test_nums = x_test.shape[0]
        losses = []
        accuracies = []
        for epoch in range(epochs):
            k = 0
            model.train()
            for batch in tqdm(range(train_nums // batch_size)):
                x = x_train[batch * batch_size : (batch + 1) * batch_size]
                y = y_train[batch * batch_size : (batch + 1) * batch_size]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                if k == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    k = acc_steps
                losses.append(loss.item())
                mlflow.log_metric("loss", loss.item())
                k -= 1
            if k > 0:
                optimizer.step()
                optimizer.zero_grad()
            total = 0
            correct = 0
            model.eval()
            for batch in tqdm(range(test_nums // batch_size)):
                x = x_test[batch * batch_size : (batch + 1) * batch_size]
                y = y_test[batch * batch_size : (batch + 1) * batch_size]
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)
                correct += (y_pred == y).sum().item()
                total += x.shape[0]
            accuracies.append(correct / total)
            mlflow.log_metric("accuracy", correct / total)
            mlflow.log_metric("epoch", epoch)
            torch.save(model.state_dict(), "../data/imdb_scorer.pth")
        mlflow.pytorch.log_model(model, "model")


def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    with mlflow.start_run():
        model, tokenizer = get_text_generator("../data/en-ru-50k.model")
        # model.load_state_dict(torch.load("../data/text_generator.pth"))
        max_sequence_length = 128
        # Train args
        scaler = torch.amp.GradScaler(device=device.type)
        num_of_sequences = 65000000
        save_every_n_steps = 10
        effective_batch_size = 1000
        batch_size = 100
        acc_steps = effective_batch_size // batch_size
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        with open("../data/OpenSubtitles.en-ru.en") as en_f, open(
            "../data/OpenSubtitles.en-ru.ru"
        ) as ru_f:
            batch_i = 0
            acc_i = 0
            save_i = 0
            seq_i = 0
            batch = torch.zeros(
                batch_size, max_sequence_length, dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            running_loss = 0.0
            ru_seq = ""
            en_seq = ""
            for en_line, ru_line in tqdm(zip(en_f, ru_f), total=num_of_sequences):
                if batch_i < batch_size:
                    tmp = False
                    if tokenizer.Encode(ru_seq)[-10] == 0:
                        ru_seq += ru_line
                        tmp = True
                    if tokenizer.Encode(en_seq)[-10] == 0:
                        en_seq += en_line
                        tmp = True
                    if tmp:
                        continue

                    batch[batch_i] = tokenizer.tokenize(ru_seq)
                    batch_i += 1
                    batch[batch_i] = tokenizer.tokenize(en_seq)
                    batch_i += 1
                    ru_seq = ""
                    en_seq = ""
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
                    torch.save(model.state_dict(), "../data/text_generator.pth")
                    save_i = 0
                seq_i += 2
                if seq_i >= num_of_sequences:
                    break
                batch_i = 0
            mlflow.pytorch.log_model(model, "model")


def train_encoder_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    with mlflow.start_run():
        model, input_tokenizer, output_tokenizer = get_translator(
            path_to_input_tokenizer="../data/en-10k.model",
            path_to_output_tokenizer="../data/ru-10k.model",
        )
        model.load_state_dict(torch.load("../data/writer.pth"))
        max_sequence_length = 16
        # Train args
        scaler = torch.amp.GradScaler(device=device.type)
        num_of_sequences = 65000000
        save_every_n_steps = 100
        effective_batch_size = 1000
        batch_size = 1000
        acc_steps = effective_batch_size // batch_size
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4, total_steps=num_of_sequences // effective_batch_size
        )
        criterion = nn.CrossEntropyLoss()
        with open("../data/OpenSubtitles.en-ru.en") as en_f, open(
            "../data/OpenSubtitles.en-ru.ru"
        ) as ru_f:
            batch_i = 0
            acc_i = 0
            save_i = 0
            seq_i = 0
            x_batch = torch.zeros(
                batch_size, max_sequence_length, dtype=torch.long, device=device
            )
            y_batch = torch.zeros(
                batch_size, max_sequence_length, dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            running_loss = 0.0
            step = 0
            for en_line, ru_line in tqdm(zip(en_f, ru_f), total=num_of_sequences):
                seq_i += 1
                if batch_i < batch_size:
                    en_seq = input_tokenizer.tokenize(en_line)
                    ru_seq = output_tokenizer.tokenize(ru_line)
                    x_batch[batch_i] = en_seq
                    y_batch[batch_i] = ru_seq
                    batch_i += 1
                    continue

                with torch.amp.autocast(device_type=device.type):
                    y_pred = model(x_batch, target=y_batch[:, :-1])
                    y_batch = y_batch[:, 1:].reshape(-1)
                    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                    loss = criterion(y_pred, y_batch)
                    step += 1

                # loss.backward()
                running_loss += loss.item()
                scaler.scale(loss).backward()
                acc_i += 1
                if acc_i == acc_steps:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    # optimizer.step()
                    optimizer.zero_grad()
                    acc_i = 0
                    save_i += 1
                if save_i == save_every_n_steps:
                    torch.save(model.state_dict(), "../data/writer.pth")
                    mlflow.log_metric(
                        "loss", running_loss / acc_steps / save_every_n_steps, step=step
                    )
                    mlflow.log_metric(
                        "learning_rate", scheduler.get_last_lr()[0], step=step
                    )
                    running_loss = 0.0
                    save_i = 0
                if seq_i >= num_of_sequences:
                    break
                batch_i = 0
                x_batch = torch.zeros(
                    batch_size, max_sequence_length, dtype=torch.long, device=device
                )
                y_batch = torch.zeros(
                    batch_size, max_sequence_length, dtype=torch.long, device=device
                )
            mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # train_encoder()
    train_decoder()
    # train_encoder_decoder()
