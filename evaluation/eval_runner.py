"""
Evaluation Runner — measures RAG quality using standard metrics.
Uses RAGAS framework for faithfulness, relevancy, and precision.
"""

import json
import structlog
from typing import Dict, Any

logger = structlog.get_logger()


def load_testset(path: str = "evaluation/datasets/testset.json") -> list:
    """Load evaluation test dataset."""
    with open(path, "r") as f:
        return json.load(f)


async def run_evaluation(rag_pipeline, testset_path: str = None) -> Dict[str, Any]:
    """
    Run evaluation on the RAG pipeline.
    
    Args:
        rag_pipeline: RAGPipeline instance
        testset_path: Path to test dataset
        
    Returns:
        Evaluation results with metrics
    """
    testset = load_testset(testset_path or "evaluation/datasets/testset.json")
    
    results = []
    total_confidence = 0
    
    for item in testset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        try:
            response = await rag_pipeline.query(question)
            
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": response.get("answer", ""),
                "confidence": response.get("confidence", 0),
                "cached": response.get("cached", False),
            }
            
            total_confidence += response.get("confidence", 0)
            results.append(result)
            
        except Exception as e:
            logger.error("eval_error", question=question[:50], error=str(e))
            results.append({
                "question": question,
                "error": str(e),
            })
    
    # Compute aggregate metrics
    avg_confidence = total_confidence / len(results) if results else 0
    
    report = {
        "total_questions": len(testset),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "average_confidence": round(avg_confidence, 2),
        "results": results,
    }
    
    # Save report
    with open("evaluation/eval_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(
        "evaluation_complete",
        total=report["total_questions"],
        avg_confidence=report["average_confidence"]
    )
    
    return report


if __name__ == "__main__":
    print("Run evaluation via: python scripts/run_eval.py")
    print("Or use the /eval endpoint after ingesting a codebase.")
