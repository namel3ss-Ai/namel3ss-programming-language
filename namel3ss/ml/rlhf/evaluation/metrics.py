"""
Evaluation metrics for RLHF training quality assessment.

This module provides a comprehensive set of metrics for evaluating RLHF-trained models:
- Reward model accuracy and correlation
- Win rate against baselines
- Output diversity and uniqueness
- Text generation quality (ROUGE, BLEU, perplexity)
- Safety metrics (toxicity, bias)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import Counter
import math


@dataclass
class MetricResult:
    """Result of a metric evaluation."""
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    samples: Optional[List[Dict[str, Any]]] = None


class EvaluationMetric(ABC):
    """
    Base class for all evaluation metrics.
    
    Metrics compute quantitative scores for model outputs,
    enabling systematic comparison of RLHF training results.
    """
    
    def __init__(self, name: str):
        """
        Initialize metric.
        
        Args:
            name: Human-readable metric name
        """
        self.name = name
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute metric on predictions.
        
        Args:
            predictions: Model-generated outputs
            references: Ground truth references (if applicable)
            **kwargs: Additional metric-specific parameters
        
        Returns:
            MetricResult with score and metadata
        """
        pass
    
    def batch_compute(
        self,
        prediction_batches: List[List[str]],
        reference_batches: Optional[List[List[str]]] = None,
        **kwargs
    ) -> List[MetricResult]:
        """
        Compute metric on multiple batches.
        
        Args:
            prediction_batches: Multiple batches of predictions
            reference_batches: Multiple batches of references
            **kwargs: Additional parameters
        
        Returns:
            List of MetricResults, one per batch
        """
        results = []
        for i, predictions in enumerate(prediction_batches):
            references = reference_batches[i] if reference_batches else None
            results.append(self.compute(predictions, references, **kwargs))
        return results


class RewardAccuracy(EvaluationMetric):
    """
    Reward model accuracy on held-out preference data.
    
    Measures how well the reward model predicts human preferences
    by comparing reward scores to ground-truth preference labels.
    """
    
    def __init__(self):
        super().__init__("reward_accuracy")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        reward_model: Optional[Any] = None,
        preference_pairs: Optional[List[Tuple[str, str, int]]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute reward model accuracy.
        
        Args:
            predictions: Not used (kept for interface compatibility)
            references: Not used
            reward_model: Reward model to evaluate
            preference_pairs: List of (chosen, rejected, label) tuples
        
        Returns:
            MetricResult with accuracy score
        """
        if not reward_model or not preference_pairs:
            return MetricResult(self.name, 0.0, {"error": "Missing reward_model or preference_pairs"})
        
        correct = 0
        total = len(preference_pairs)
        
        for chosen, rejected, label in preference_pairs:
            # Get reward scores
            chosen_reward = reward_model.score(chosen)
            rejected_reward = reward_model.score(rejected)
            
            # Check if reward ordering matches preference
            if (chosen_reward > rejected_reward and label == 1) or \
               (chosen_reward < rejected_reward and label == 0):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return MetricResult(
            self.name,
            accuracy,
            metadata={
                "correct": correct,
                "total": total,
                "error_rate": 1.0 - accuracy
            }
        )


class WinRate(EvaluationMetric):
    """
    Win rate against baseline model.
    
    Measures percentage of times the RLHF model output is preferred
    over a baseline model output by evaluators (human or automatic).
    """
    
    def __init__(self):
        super().__init__("win_rate")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        baseline_predictions: Optional[List[str]] = None,
        preference_labels: Optional[List[int]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute win rate.
        
        Args:
            predictions: RLHF model outputs
            references: Not used
            baseline_predictions: Baseline model outputs
            preference_labels: 1 if RLHF preferred, 0 if baseline, 0.5 if tie
        
        Returns:
            MetricResult with win rate
        """
        if not baseline_predictions or not preference_labels:
            return MetricResult(self.name, 0.0, {"error": "Missing baseline_predictions or preference_labels"})
        
        if len(predictions) != len(baseline_predictions) or len(predictions) != len(preference_labels):
            return MetricResult(self.name, 0.0, {"error": "Length mismatch"})
        
        wins = sum(1 for label in preference_labels if label == 1)
        ties = sum(1 for label in preference_labels if label == 0.5)
        losses = sum(1 for label in preference_labels if label == 0)
        total = len(preference_labels)
        
        win_rate = wins / total if total > 0 else 0.0
        
        return MetricResult(
            self.name,
            win_rate,
            metadata={
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "total": total,
                "win_tie_rate": (wins + ties) / total if total > 0 else 0.0
            }
        )


class Diversity(EvaluationMetric):
    """
    Output diversity metric.
    
    Measures lexical and semantic diversity of model outputs
    to detect mode collapse or repetitive generation.
    """
    
    def __init__(self):
        super().__init__("diversity")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute diversity metrics.
        
        Args:
            predictions: Model outputs to analyze
            references: Not used
        
        Returns:
            MetricResult with diversity score (0-1, higher is better)
        """
        if not predictions:
            return MetricResult(self.name, 0.0, {"error": "No predictions"})
        
        # Compute distinct n-gram ratios
        distinct_1 = self._distinct_ngrams(predictions, n=1)
        distinct_2 = self._distinct_ngrams(predictions, n=2)
        distinct_3 = self._distinct_ngrams(predictions, n=3)
        
        # Compute vocabulary richness
        all_words = []
        for pred in predictions:
            all_words.extend(pred.split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        vocab_richness = unique_words / total_words if total_words > 0 else 0.0
        
        # Average diversity score
        diversity_score = (distinct_1 + distinct_2 + distinct_3 + vocab_richness) / 4.0
        
        return MetricResult(
            self.name,
            diversity_score,
            metadata={
                "distinct_1": distinct_1,
                "distinct_2": distinct_2,
                "distinct_3": distinct_3,
                "vocab_richness": vocab_richness,
                "unique_words": unique_words,
                "total_words": total_words
            }
        )
    
    def _distinct_ngrams(self, texts: List[str], n: int) -> float:
        """Compute distinct n-gram ratio."""
        all_ngrams = []
        for text in texts:
            words = text.split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        return len(set(all_ngrams)) / len(all_ngrams)


class Perplexity(EvaluationMetric):
    """
    Language model perplexity.
    
    Measures how well the model predicts a sample text.
    Lower perplexity indicates better language modeling.
    """
    
    def __init__(self):
        super().__init__("perplexity")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute perplexity.
        
        Args:
            predictions: Texts to evaluate
            references: Not used
            model: Language model with log_prob method
        
        Returns:
            MetricResult with perplexity score
        """
        if not model:
            return MetricResult(self.name, float('inf'), {"error": "No model provided"})
        
        total_log_prob = 0.0
        total_tokens = 0
        
        for text in predictions:
            # Get log probability from model
            log_prob = model.log_prob(text)
            tokens = len(text.split())
            
            total_log_prob += log_prob
            total_tokens += tokens
        
        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(-avg_log_prob)
        
        return MetricResult(
            self.name,
            perplexity,
            metadata={
                "avg_log_prob": avg_log_prob,
                "total_tokens": total_tokens
            }
        )


class RougeScore(EvaluationMetric):
    """
    ROUGE score for text generation quality.
    
    Measures overlap of n-grams between generated and reference texts.
    Commonly used for summarization and generation tasks.
    """
    
    def __init__(self, rouge_type: str = "rouge-l"):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_type: Type of ROUGE ('rouge-1', 'rouge-2', 'rouge-l')
        """
        super().__init__(f"rouge_{rouge_type}")
        self.rouge_type = rouge_type
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute ROUGE score.
        
        Args:
            predictions: Generated texts
            references: Reference texts
        
        Returns:
            MetricResult with ROUGE F1 score
        """
        if not references:
            return MetricResult(self.name, 0.0, {"error": "No references provided"})
        
        if len(predictions) != len(references):
            return MetricResult(self.name, 0.0, {"error": "Length mismatch"})
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self._compute_rouge(pred, ref)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        return MetricResult(
            self.name,
            avg_score,
            metadata={
                "scores": scores,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "std": np.std(scores) if scores else 0.0
            }
        )
    
    def _compute_rouge(self, prediction: str, reference: str) -> float:
        """Compute ROUGE score for a single pair."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if self.rouge_type == "rouge-1":
            return self._rouge_n(pred_tokens, ref_tokens, n=1)
        elif self.rouge_type == "rouge-2":
            return self._rouge_n(pred_tokens, ref_tokens, n=2)
        elif self.rouge_type == "rouge-l":
            return self._rouge_l(pred_tokens, ref_tokens)
        else:
            return 0.0
    
    def _rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute ROUGE-N score."""
        pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        
        if not ref_ngrams:
            return 0.0
        
        recall = overlap / sum(ref_ngrams.values())
        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
        
        if recall + precision == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L score using longest common subsequence."""
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if not ref_tokens or not pred_tokens:
            return 0.0
        
        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(pred_tokens)
        
        if recall + precision == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class BLEUScore(EvaluationMetric):
    """
    BLEU score for machine translation quality.
    
    Measures n-gram precision between generated and reference texts.
    Originally designed for translation, also used for generation tasks.
    """
    
    def __init__(self, max_n: int = 4):
        super().__init__("bleu")
        self.max_n = max_n
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute BLEU score.
        
        Args:
            predictions: Generated texts
            references: Reference texts
        
        Returns:
            MetricResult with BLEU score
        """
        if not references:
            return MetricResult(self.name, 0.0, {"error": "No references provided"})
        
        if len(predictions) != len(references):
            return MetricResult(self.name, 0.0, {"error": "Length mismatch"})
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self._compute_bleu(pred, ref)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        return MetricResult(
            self.name,
            avg_score,
            metadata={
                "scores": scores,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0
            }
        )
    
    def _compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score for a single pair."""
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)])
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            precision = overlap / total if total > 0 else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if not precisions or any(p == 0 for p in precisions):
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        bleu = bp * geometric_mean
        return bleu


class ToxicityScore(EvaluationMetric):
    """
    Toxicity detection metric.
    
    Measures toxicity, profanity, and harmful content in model outputs
    using rule-based filters or ML-based detectors (Perspective API).
    """
    
    def __init__(self):
        super().__init__("toxicity")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        detector: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute toxicity scores.
        
        Args:
            predictions: Texts to evaluate
            references: Not used
            detector: Toxicity detector with score() method
        
        Returns:
            MetricResult with average toxicity (0-1, lower is better)
        """
        if not detector:
            # Fallback to simple rule-based detection
            toxic_words = {'hate', 'kill', 'stupid', 'idiot', 'damn', 'hell'}
            scores = []
            for text in predictions:
                words = set(text.lower().split())
                toxicity = len(words & toxic_words) / len(words) if words else 0.0
                scores.append(min(toxicity, 1.0))
        else:
            scores = [detector.score(text) for text in predictions]
        
        avg_toxicity = np.mean(scores) if scores else 0.0
        max_toxicity = max(scores) if scores else 0.0
        toxic_count = sum(1 for s in scores if s > 0.5)
        
        return MetricResult(
            self.name,
            avg_toxicity,
            metadata={
                "max_toxicity": max_toxicity,
                "toxic_count": toxic_count,
                "toxic_rate": toxic_count / len(scores) if scores else 0.0,
                "scores": scores
            }
        )


class BiasScore(EvaluationMetric):
    """
    Demographic bias detection metric.
    
    Measures potential biases in model outputs related to
    gender, race, religion, and other protected attributes.
    """
    
    def __init__(self):
        super().__init__("bias")
    
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        bias_detector: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute bias scores.
        
        Args:
            predictions: Texts to evaluate
            references: Not used
            bias_detector: Bias detector with score() method
        
        Returns:
            MetricResult with average bias score (0-1, lower is better)
        """
        if not bias_detector:
            # Placeholder implementation
            return MetricResult(
                self.name,
                0.0,
                metadata={"error": "No bias detector provided"}
            )
        
        scores = []
        bias_types = {"gender": [], "race": [], "religion": [], "age": []}
        
        for text in predictions:
            result = bias_detector.score(text)
            scores.append(result.get("overall", 0.0))
            
            for bias_type in bias_types:
                if bias_type in result:
                    bias_types[bias_type].append(result[bias_type])
        
        avg_bias = np.mean(scores) if scores else 0.0
        
        metadata = {
            "overall_bias": avg_bias,
            "max_bias": max(scores) if scores else 0.0
        }
        
        for bias_type, type_scores in bias_types.items():
            if type_scores:
                metadata[f"{bias_type}_bias"] = np.mean(type_scores)
        
        return MetricResult(self.name, avg_bias, metadata=metadata)


__all__ = [
    "EvaluationMetric",
    "MetricResult",
    "RewardAccuracy",
    "WinRate",
    "Diversity",
    "Perplexity",
    "RougeScore",
    "BLEUScore",
    "ToxicityScore",
    "BiasScore",
]
