import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased',
  output_hidden_states = True,
  )

model.eval()


def sent2vec(s):
    t = f'[CLS] {s} [SEP]'
    tok = tokenizer.tokenize(t)
    ids = tokenizer.convert_tokens_to_ids(tok)
    segs = [1] * len(tok)

    tok_t = torch.tensor([ids])
    seg_t = torch.tensor([segs])

    with torch.no_grad():
        out = model(tok_t, seg_t)

    hid = out[2]
    return torch.mean(hid[-2][0], dim=0)


def embed(s):
    # TODO: split sentences and find mean
    return sent2vec(s)


def cmp(v1, v2):
    return 1 - cosine(v1, v2)


def sim(s1, s2):
    return cmp(sent2vec(s1), sent2vec(s2))

