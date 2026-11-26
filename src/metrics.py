from sacrebleu.metrics import BLEU, CHRF


def metrics(rows):
    scores = {}
    for direction in sorted({row["direction"] for row in rows}):
        group = [row for row in rows if row["direction"] == direction]
        predictions = [row["translation"] for row in group]
        references = [[row["reference"] for row in group]]
        bleu, chrf = BLEU(tokenize="13a"), CHRF(word_order=2)
        scores[direction] = {
            "count": len(group),
            "bleu": bleu.corpus_score(predictions, references).score,
            "chrf++": chrf.corpus_score(predictions, references).score,
            "bleu_signature": str(bleu.get_signature()),
            "chrf++_signature": str(chrf.get_signature()),
        }
    return scores
