from enum import Enum, unique

import diskcache
import torch
from transformers import AutoTokenizer, BertModel
from scipy.spatial.distance import cosine
from spacy.lang.en import English

import common


cache = diskcache.Cache(".simcache")

nlp = English()
nlp.add_pipe('sentencizer')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased',
  output_hidden_states = True,
  )

model.eval()


@unique
class PoolingStrategy(Enum):
    """Strategies for pooling BERT word embeddings to a chunk of text."""
    # Take the mean across all penultimate layers of all words in a sentence,
    # then all sentences.
    PENULT = 'penult'
    # Take the mean of the last four layers of all words in a sentence,
    # then all sentences.
    AVG4 = 'avg4'


def tokenize(s, split_sentences=False, max_length=128):
    """Tokenize a given chunk of text as BERT model inputs.

    Returns:
        Inputs for BERT model. If the input text contained multiple sentences,
        they will be included individually so that the model runs a batch.
    """
    batch = s
    if split_sentences:
        doc = nlp(s)
        batch = [str(s) for s in doc.sents]

    return tokenizer(batch,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')


@cache.memoize()
def run_model(s):
    """Get an embedding for a span of text.

    Text is sentence-tokenized, passed to BERT, and the resulting word
    embeddings are pooled using the given strategy.

    Returns:
        Model output
    """
    inputs = tokenize(s)

    with torch.no_grad():
        out = model(**inputs)

    return out


def pool(model_out, strategy=PoolingStrategy.PENULT):
    """Pool the embedding layers to come up with a single dimension vector.

    Returns:
        One dimensional tensor of size 1024.
    """
    # Get the hidden states of the batch of inputs.
    # Shape: len(inputs) x 512 x 1024
    hid = model_out[2]

    if strategy == PoolingStrategy.PENULT:
        # Pool the penultimate hidden layer across all sentences.
        return torch.mean(hid[-2], dim=(0, 1))

    if strategy == PoolingStrategy.AVG4:
        # Pool the last four hidden layers in all sentences.
        return torch.mean(torch.stack(hid[-4:]), dim=(0, 1, 2))

    raise ValueError(f"Unknown pooling strategy: {strategy}")


def embed(s, **kwargs):
    """Get an embedding for a span of text.

    Runs the model and pools the resulting word embeddings using the given
    pooling strategy.

    Returns:
        One-dimensional tensor of size 1024.
    """
    pool_args = common.filter_args({'strategy'}, kwargs)
    return pool(run_model(s), **pool_args)


def cmp(v1, v2):
    """Compute distance between two embeddings using cosine similarity.

    Returns:
        Float of the similarity (higher is more similar).
    """
    return 1 - cosine(v1, v2)


def sim(s1, s2, **kwargs):
    """Compute similarity between two chunks of text.

    Returns:
        Float of similarity (higher is more similar).
    """
    return cmp(embed(s1, **kwargs), embed(s2, **kwargs))

