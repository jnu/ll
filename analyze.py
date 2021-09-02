import diskcache

import sim
import prof


cache = diskcache.Cache(".simcache")


@cache.memoize()
def embeddings_for_prof(path):
    qhist = prof.load(path)
    return [
        [s, outcome, sim.embed(s)]
        for s, outcome in qhist]


def find_similar(embeddings, q):
    qembed = sim.embed(q)

    compared = [
        [s, outcome, sim.cmp(embedding, qembed)]
        for s, outcome, embedding in embeddings]

    compared.sort(reverse=True, key=lambda r: r[2])

    return compared


def predict(similar):
    return sum(outcome * similarity for s, outcome, similarity in similar) / float(len(similar))


def analyze(path, q, n=10):
    embeddings = embeddings_for_prof(path)
    similar = find_similar(embeddings, q)

    print("Most similar questions:")
    for s, outcome, similarity in similar[:n]:
        print(" -", s, outcome, similarity)
    
    p = predict(similar)
    print("Correct answer confidence:", p)
