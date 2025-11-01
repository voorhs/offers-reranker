"""Evaluation metrics for ranking: NDCG and MRR."""

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at position k.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Position to compute DCG at

    Returns:
        DCG@k score
    """
    relevance_scores = np.array(relevance_scores[:k])
    if relevance_scores.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return float(np.sum(relevance_scores / discounts))


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at position k.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Position to compute NDCG at

    Returns:
        NDCG@k score (0-1)
    """
    dcg = dcg_at_k(relevance_scores, k)

    # Ideal DCG: sort scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(relevance_scores: list[float], threshold: float = 50.0) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        relevance_scores: List of relevance scores in ranked order
        threshold: Minimum score to consider relevant

    Returns:
        Reciprocal rank (0-1), or 0 if no relevant items
    """
    for i, score in enumerate(relevance_scores, start=1):
        if score >= threshold:
            return 1.0 / i
    return 0.0


class CrossEncoderRankingEvaluator(SentenceEvaluator):
    """Evaluator for ranking metrics: NDCG and MRR.

    This evaluator computes ranking metrics by:
    1. Scoring all query-offer pairs
    2. Ranking offers by predicted scores
    3. Computing NDCG@k and MRR using gold scores
    """

    def __init__(
        self,
        eval_data: list[dict],
        name: str = "eval",
        ndcg_at_k: list[int] | None = None,
        mrr_threshold: float = 50.0,
    ):
        """Initialize evaluator.

        Args:
            eval_data: List of dicts with keys:
                - "query": str
                - "texts": list[str] (offers)
                - "scores": list[float] (gold relevance scores 0-100)
            name: Name prefix for metrics
            ndcg_at_k: List of k values for NDCG computation (default: [1, 3, 5])
            mrr_threshold: Threshold for considering items relevant in MRR
        """
        self.eval_data = eval_data
        self.name = name
        self.ndcg_at_k = ndcg_at_k or [1, 3, 5]
        self.mrr_threshold = mrr_threshold

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """Evaluate model and return primary metric.

        Args:
            model: CrossEncoder model to evaluate
            output_path: Path to save results (optional)
            epoch: Current epoch number
            steps: Current step number

        Returns:
            Primary metric value (NDCG@3)
        """
        ndcg_scores = {k: [] for k in self.ndcg_at_k}
        mrr_scores = []

        for sample in self.eval_data:
            query = sample["query"]
            texts = sample["texts"]
            gold_scores = sample["scores"]

            if not texts:
                continue

            # Create pairs for scoring
            pairs = [[query, text] for text in texts]

            # Predict scores
            predicted_scores = model.predict(pairs, convert_to_numpy=True)

            # Rank offers by predicted scores (descending)
            ranked_indices = np.argsort(predicted_scores)[::-1]
            ranked_gold_scores = [gold_scores[i] for i in ranked_indices]

            # Compute NDCG@k for different k values
            for k in self.ndcg_at_k:
                ndcg_scores[k].append(ndcg_at_k(ranked_gold_scores, k))

            # Compute MRR
            mrr_scores.append(mrr(ranked_gold_scores, threshold=self.mrr_threshold))

        # Average metrics across all queries
        metrics = {}
        for k in self.ndcg_at_k:
            metrics[f"{self.name}_ndcg@{k}"] = float(np.mean(ndcg_scores[k]))

        metrics[f"{self.name}_mrr"] = float(np.mean(mrr_scores))

        # Log metrics
        if epoch != -1:
            logger.info(f"\nEpoch {epoch}:")
        elif steps != -1:
            logger.info(f"\nStep {steps}:")

        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Store metrics for retrieval
        self.last_metrics = metrics

        # Return primary metric (NDCG@3)
        return metrics[f"{self.name}_ndcg@3"]
