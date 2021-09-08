import sim
import prof
import match
import common



def embeddings_for_prof(path, **kwargs):
    qhist = prof.load(path)
    return [(q, sim.embed(q.text, **kwargs)) for q in qhist]


def find_similar(embeddings, q, cutoff=0.5):
    qembed = sim.embed(q)

    compared = [(q, sim.cmp(embedding, qembed)) for q, embedding in embeddings]

    compared.sort(reverse=True, key=lambda c: c[1])

    # Drop questions below the cutoff and those that match exactly
    return [c for c in compared if c[1] >= cutoff and c[1] != 1.]


def predict(similar):
    if not similar:
        return 0.

    # Average of outcomes weighted by similarity
    num = sum(q.correct * similarity for q, similarity in similar)
    denom = sum(similarity for _, similarity in similar)
    return num / denom


def analyze_question(path, q, **kwargs):
    embargs = common.filter_args({'strategy'}, kwargs)
    embeddings = embeddings_for_prof(path, **embargs)

    simargs = common.filter_args({'cutoff'}, kwargs)
    similar = find_similar(embeddings, q, **simargs)

    print(f"\nMost similar questions (n={len(similar)}):\n")
    for q, similarity in similar:
        print(" -", q.text, q.correct, similarity)
    
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
        print(f"{i + 1}. {q}")
        print("Points: {} (confidence={:.3f})".format(pt, conf))
        print("")

    return predictions
