import random
import time
import datetime

import numpy as np
import torch
from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler,
        SequentialSampler,
        random_split,
        )
from transformers import (
        AutoTokenizer,
        BertForSequenceClassification,
        AdamW,
        BertConfig,
        get_linear_schedule_with_warmup,
        )

import prof



tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')


def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def tokenize(s, max_length=128):
    return tokenizer(s,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')


def dataset_for_prof(path):
    qhist = prof.load(path)

    tokenized = [tokenize(q.text) for q in qhist]

    input_ids = torch.cat([t['input_ids'] for t in tokenized], dim=0)
    attention_masks = torch.cat([t['attention_mask'] for t in tokenized], dim=0)
    labels = torch.tensor([int(q.correct) for q in qhist])

    return TensorDataset(input_ids, attention_masks, labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(model):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Find a free GPU
            if torch.cuda.memory_reserved(i) > 0:
                continue
            print("Using GPU {} for training: {}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda:{}'.format(i))
            break
        model.cuda()
    else:
        print("No GPU available; training on CPU.")
        device = torch.device('cpu')
    return device


def train_on_prof(path, batch_size=32, holdback=0.1, learn_rate=2e-5, epsilon=1e-8, epochs=4, seed=1207):
    ds = dataset_for_prof(path)

    n_train = int((1. - holdback) * len(ds))
    n_test = len(ds) - n_train

    train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_dl = DataLoader(
            train_ds,
            sampler=RandomSampler(train_ds),
            batch_size=batch_size,
            )
    
    test_dl = DataLoader(
            test_ds,
            sampler=SequentialSampler(test_ds),
            batch_size=batch_size,
            )

    model = BertForSequenceClassification.from_pretrained('bert-large-uncased',
            output_attentions=True,
            output_hidden_states=True,
            num_labels=2,
            )

    device = get_device(model)

    optimizer = AdamW(model.parameters(),
            lr=learn_rate,
            eps=epsilon,
            )

    total_steps = len(train_dl) * epochs
    print("Total steps:", total_steps)

    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
            )

    # == TRAIN ==
    set_seed(seed)

    all_stats = []
    t0 = time.time()

    for epx in range(epochs):
        print("\n=== EPOCH {:} / {:} ===\n".format(epx + 1, epochs))
        stats = {'epoch': epx + 1}

        stats.update(
                train_epoch(device, model, scheduler, optimizer, train_dl),
                )
        print("")
        stats.update(
                eval_epoch(device, model, test_dl),
                )

        all_stats.append(stats)

    print("")
    print("Done training!")
    print("Total time:", format_time(time.time() - t0))

    return model, all_stats


def eval_epoch(device, model, dataloader):
    print("Running validation ...")
    
    t0 = time.time()
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_attention_mask,
                    labels=b_labels,
                    )
        loss = output['loss']
        logits = output['logits']

        total_eval_loss += loss.item()

        # Move stuff back to the CPU to compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += accuracy(logits, label_ids)

    avg_accuracy = total_eval_accuracy / len(dataloader)
    avg_loss = total_eval_loss / len(dataloader)
    elapsed_t = format_time(time.time() - t0)

    print("  Accuracy: {0:.2f}".format(avg_accuracy))
    print("  Validation loss: {0:.2f}".format(avg_loss))
    print("  Validation time: {:}".format(elapsed_t))

    return {
            'Validation Loss': avg_loss,
            'Validation Accuracy': avg_accuracy,
            'Validation Time': elapsed_t,
            }


def format_time(elapsed):
    rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=rounded))


def train_epoch(device, model, scheduler, optimizer, dataloader):
    print("Training ...")
    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(dataloader):
        # Print progress sometimes
        if step % 40 == 0 and step > 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        # Get batch parameters
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear gradient
        model.zero_grad()

        # Compute forward pass
        output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_attention_mask,
                labels=b_labels,
                )

        loss = output['loss']

        # Add loss
        total_train_loss += loss.item()

        # Perform backward pass
        loss.backward()

        # Clip the norm of the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()
        scheduler.step()

    # Calculate stats
    avg_train_loss = total_train_loss / len(dataloader)
    elapsed_t = format_time(time.time() - t0)
    print("  Training loss: {0:.2f}".format(avg_train_loss))
    print("  Elapsed time for epoch: {:}".format(elapsed_t))

    return {
            'Training Loss': avg_train_loss,
            'Training Time': elapsed_t,
            }
