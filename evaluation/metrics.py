"""
Evaluation Metrics — custom metrics for RAG quality assessment.
"""


def answer_similarity(generated: str, reference: str) -> float:
    """
    Simple word overlap similarity between generated and reference answers.
    Production: use embedding similarity instead.
    """
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = gen_words.intersection(ref_words)
    return len(overlap) / len(ref_words)


def answer_completeness(generated: str, reference: str) -> float:
    """
    Check if key concepts from reference appear in generated answer.
    """
    ref_words = set(reference.lower().split())
    gen_words = set(generated.lower().split())
    
    # Filter to meaningful words (> 3 chars)
    key_words = {w for w in ref_words if len(w) > 3}
    
    if not key_words:
        return 1.0
    
    found = key_words.intersection(gen_words)
    return len(found) / len(key_words)


def compute_metrics(results: list) -> dict:
    """
    Compute aggregate metrics from evaluation results.
    """
    similarities = []
    completeness_scores = []
    
    for result in results:
        if "error" in result:
            continue
        
        sim = answer_similarity(
            result.get("answer", ""),
            result.get("ground_truth", "")
        )
        comp = answer_completeness(
            result.get("answer", ""),
            result.get("ground_truth", "")
        )
        
        similarities.append(sim)
        completeness_scores.append(comp)
    
    return {
        "answer_similarity": round(
            sum(similarities) / len(similarities), 4
        ) if similarities else 0,
        "answer_completeness": round(
            sum(completeness_scores) / len(completeness_scores), 4
        ) if completeness_scores else 0,
        "total_evaluated": len(similarities),
    }
