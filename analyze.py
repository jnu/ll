import diskcache

import sim
import prof
import match


cache = diskcache.Cache(".simcache")


@cache.memoize()
def embeddings_for_prof(path):
    qhist = prof.load(path)
    return [
        [s, outcome, sim.embed(s)]
        for s, outcome in qhist]


def find_similar(embeddings, q, cutoff=0.5):
    qembed = sim.embed(q)

    compared = [
        [s, outcome, sim.cmp(embedding, qembed)]
        for s, outcome, embedding in embeddings]

    compared.sort(reverse=True, key=lambda r: r[2])

    return [c for c in compared if c[2] >= cutoff]


def predict(similar):
    if not similar:
        return 0.

    return sum(outcome * similarity for s, outcome, similarity in similar) / float(len(similar))


def analyze_question(path, q, cutoff=0.75):
    embeddings = embeddings_for_prof(path)
    similar = find_similar(embeddings, q, cutoff=cutoff)

    print(f"Most similar questions (cutoff={cutoff},n={len(similar)}):")
    for s, outcome, similarity in similar:
        print(" -", s, outcome, similarity)
    
    p = predict(similar)
    print("Correct answer confidence:", p)

    return p


def defend(history_path, match_path, **kwargs):
    qs = match.load(match_path)

    # Predict outcome based on similar historical questions
    raw_predictions = [[i, q, analyze_question(history_path, q, **kwargs)] for i, q in enumerate(qs)]

    # Sort by most confident to least
    raw_predictions.sort(reverse=True, key=lambda x: x[2])

    # Assign points in order of confidence (fewest points to most confident)
    points = [0, 1, 1, 2, 2, 3]
    predictions = [[i, q, conf, points[x]] for x, (i, q, conf) in enumerate(raw_predictions)]

    # Put back in original order
    predictions.sort(key=lambda x: x[0])
    
    print("\n--- DEFENSE ---\n")
    for i, q, conf, pt in predictions:
        print(f"{i + 1}. {q} {pt} ({conf})")

    return predictions
