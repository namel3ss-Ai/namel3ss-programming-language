"""
Unit tests for RLHF evaluation metrics.

Tests all 9 metric implementations with synthetic data:
- RewardAccuracy
- WinRate
- Diversity
- Perplexity
- RougeScore
- BLEUScore
- ToxicityScore
- BiasScore
"""

import pytest
import math
from typing import List
from namel3ss.ml.rlhf.evaluation.metrics import (
    RewardAccuracy,
    WinRate,
    Diversity,
    Perplexity,
    RougeScore,
    BLEUScore,
    ToxicityScore,
    BiasScore,
    MetricResult,
)


class TestRewardAccuracy:
    """Test reward model accuracy metric."""

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        metric = RewardAccuracy()
        
        predictions = [
            {"chosen_reward": 0.9, "rejected_reward": 0.2},
            {"chosen_reward": 0.8, "rejected_reward": 0.3},
            {"chosen_reward": 0.95, "rejected_reward": 0.1},
        ]
        references = [
            {"preference": "chosen"},
            {"preference": "chosen"},
            {"preference": "chosen"},
        ]
        
        result = metric.compute(predictions, references)
        
        assert result.name == "reward_accuracy"
        assert result.value == 1.0
        assert result.metadata["correct"] == 3
        assert result.metadata["total"] == 3
        assert result.metadata["error_rate"] == 0.0

    def test_mixed_accuracy(self):
        """Test with mixed correct/incorrect predictions."""
        metric = RewardAccuracy()
        
        predictions = [
            {"chosen_reward": 0.9, "rejected_reward": 0.2},  # Correct
            {"chosen_reward": 0.3, "rejected_reward": 0.8},  # Correct (rejected preferred)
            {"chosen_reward": 0.4, "rejected_reward": 0.9},  # Correct (rejected preferred)
            {"chosen_reward": 0.2, "rejected_reward": 0.7},  # Incorrect (should prefer chosen)
        ]
        references = [
            {"preference": "chosen"},
            {"preference": "rejected"},
            {"preference": "rejected"},
            {"preference": "chosen"},
        ]
        
        result = metric.compute(predictions, references)
        
        assert result.value == 0.75  # 3 out of 4 correct
        assert result.metadata["correct"] == 3
        assert result.metadata["total"] == 4
        assert result.metadata["error_rate"] == 0.25

    def test_zero_accuracy(self):
        """Test with no correct predictions."""
        metric = RewardAccuracy()
        
        predictions = [
            {"chosen_reward": 0.2, "rejected_reward": 0.9},
            {"chosen_reward": 0.1, "rejected_reward": 0.8},
        ]
        references = [
            {"preference": "chosen"},
            {"preference": "chosen"},
        ]
        
        result = metric.compute(predictions, references)
        
        assert result.value == 0.0
        assert result.metadata["correct"] == 0
        assert result.metadata["total"] == 2

    def test_batch_compute(self):
        """Test batch computation."""
        metric = RewardAccuracy()
        
        batches = [
            (
                [{"chosen_reward": 0.9, "rejected_reward": 0.2}],
                [{"preference": "chosen"}]
            ),
            (
                [{"chosen_reward": 0.3, "rejected_reward": 0.8}],
                [{"preference": "rejected"}]
            ),
        ]
        
        results = metric.batch_compute(batches)
        
        assert len(results) == 2
        assert all(r.value == 1.0 for r in results)


class TestWinRate:
    """Test win rate comparison metric."""

    def test_all_wins(self):
        """Test when RLHF model wins all comparisons."""
        metric = WinRate()
        
        predictions = ["The RLHF model is better."] * 3
        references = ["The baseline is worse."] * 3
        
        result = metric.compute(predictions, references)
        
        assert result.name == "win_rate"
        assert result.value == 1.0
        assert result.metadata["wins"] == 3
        assert result.metadata["ties"] == 0
        assert result.metadata["losses"] == 0

    def test_mixed_outcomes(self):
        """Test with mixed wins/ties/losses."""
        metric = WinRate()
        
        # Simulate judgments based on response length as proxy
        predictions = [
            "Excellent detailed response here.",  # Win
            "Okay response.",  # Tie
            "Bad.",  # Loss
            "Another great detailed response.",  # Win
        ]
        references = [
            "Mediocre response.",
            "Okay response here.",
            "Much better response than that.",
            "Poor response.",
        ]
        
        result = metric.compute(predictions, references)
        
        assert 0.0 <= result.value <= 1.0
        assert result.metadata["wins"] + result.metadata["ties"] + result.metadata["losses"] == 4

    def test_win_tie_rate(self):
        """Test win+tie rate calculation."""
        metric = WinRate()
        
        predictions = ["Good"] * 5
        references = ["Bad"] * 5
        
        result = metric.compute(predictions, references)
        
        win_tie_rate = result.metadata["win_tie_rate"]
        assert 0.0 <= win_tie_rate <= 1.0


class TestDiversity:
    """Test diversity metric."""

    def test_high_diversity(self):
        """Test with highly diverse text."""
        metric = Diversity()
        
        predictions = [
            "The quick brown fox jumps over the lazy dog.",
            "A completely different sentence with unique words.",
            "Yet another distinct phrase without repetition.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.name == "diversity"
        assert result.value > 0.5  # High diversity expected
        assert "distinct_1grams" in result.metadata
        assert "distinct_2grams" in result.metadata
        assert "distinct_3grams" in result.metadata
        assert "vocab_richness" in result.metadata

    def test_low_diversity(self):
        """Test with repetitive text."""
        metric = Diversity()
        
        predictions = [
            "hello hello hello hello",
            "hello hello hello hello",
            "hello hello hello hello",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value < 0.5  # Low diversity expected
        assert result.metadata["distinct_1grams"] < 0.2  # Very few unique unigrams

    def test_vocab_richness(self):
        """Test vocabulary richness calculation."""
        metric = Diversity()
        
        predictions = [
            "a b c d e f g h i j k l m n o p",  # 16 unique words
        ]
        
        result = metric.compute(predictions)
        
        # Vocab richness = unique_words / total_words = 16/16 = 1.0
        assert result.metadata["vocab_richness"] == 1.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        metric = Diversity()
        
        predictions = ["", ""]
        
        result = metric.compute(predictions)
        
        assert result.value == 0.0


class TestPerplexity:
    """Test perplexity metric."""

    def test_low_perplexity(self):
        """Test with low perplexity (good model)."""
        metric = Perplexity()
        
        # High log probabilities -> low perplexity
        predictions = [
            {"log_probs": [-0.1, -0.2, -0.15]},
            {"log_probs": [-0.12, -0.18, -0.11]},
        ]
        
        result = metric.compute(predictions)
        
        assert result.name == "perplexity"
        assert result.value < 2.0  # Low perplexity

    def test_high_perplexity(self):
        """Test with high perplexity (poor model)."""
        metric = Perplexity()
        
        # Low log probabilities -> high perplexity
        predictions = [
            {"log_probs": [-5.0, -6.0, -5.5]},
            {"log_probs": [-5.2, -5.8, -6.1]},
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 100.0  # High perplexity

    def test_perplexity_calculation(self):
        """Test mathematical correctness of perplexity."""
        metric = Perplexity()
        
        # Known values: log_prob = -ln(2) means prob = 0.5
        # Perplexity = exp(-avg_log_prob) = exp(ln(2)) = 2
        log_2 = -math.log(0.5)
        
        predictions = [
            {"log_probs": [log_2, log_2, log_2]},
        ]
        
        result = metric.compute(predictions)
        
        assert abs(result.value - 2.0) < 0.01  # Should be ~2.0


class TestRougeScore:
    """Test ROUGE metric."""

    def test_rouge_1_perfect_match(self):
        """Test ROUGE-1 with perfect match."""
        metric = RougeScore(rouge_type="rouge1")
        
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        result = metric.compute(predictions, references)
        
        assert result.name == "rouge1"
        assert result.value == 1.0  # Perfect match

    def test_rouge_1_partial_match(self):
        """Test ROUGE-1 with partial overlap."""
        metric = RougeScore(rouge_type="rouge1")
        
        predictions = ["the cat sat"]
        references = ["the dog sat"]
        
        result = metric.compute(predictions, references)
        
        # "the" and "sat" overlap (2 out of 3)
        assert 0.5 < result.value < 1.0

    def test_rouge_2_bigram_overlap(self):
        """Test ROUGE-2 bigram overlap."""
        metric = RougeScore(rouge_type="rouge2")
        
        predictions = ["the cat sat on"]
        references = ["the cat sat down"]
        
        result = metric.compute(predictions, references)
        
        # Bigrams: "the cat", "cat sat" overlap (2 out of 3)
        assert result.value > 0.5

    def test_rouge_l_longest_common_subsequence(self):
        """Test ROUGE-L with LCS."""
        metric = RougeScore(rouge_type="rougeL")
        
        predictions = ["the cat sat"]
        references = ["the big cat sat down"]
        
        result = metric.compute(predictions, references)
        
        # LCS is "the cat sat" (3 words)
        assert result.value > 0.5

    def test_rouge_no_overlap(self):
        """Test ROUGE with no overlap."""
        metric = RougeScore(rouge_type="rouge1")
        
        predictions = ["foo bar baz"]
        references = ["qux quux corge"]
        
        result = metric.compute(predictions, references)
        
        assert result.value == 0.0


class TestBLEUScore:
    """Test BLEU metric."""

    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match."""
        metric = BLEUScore(n=4)
        
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        result = metric.compute(predictions, references)
        
        assert result.name == "bleu"
        assert result.value == 1.0  # Perfect match

    def test_bleu_partial_match(self):
        """Test BLEU with partial match."""
        metric = BLEUScore(n=2)
        
        predictions = ["the cat sat"]
        references = ["the dog sat"]
        
        result = metric.compute(predictions, references)
        
        assert 0.3 < result.value < 0.9  # Partial match

    def test_bleu_brevity_penalty(self):
        """Test BLEU brevity penalty for short predictions."""
        metric = BLEUScore(n=2)
        
        predictions = ["cat"]  # Very short
        references = ["the cat sat on the mat"]  # Much longer
        
        result = metric.compute(predictions, references)
        
        # Should have low score due to brevity penalty
        assert result.value < 0.5
        assert "brevity_penalty" in result.metadata
        assert result.metadata["brevity_penalty"] < 1.0

    def test_bleu_no_brevity_penalty(self):
        """Test BLEU without brevity penalty (pred >= ref length)."""
        metric = BLEUScore(n=2)
        
        predictions = ["the cat sat on the mat and more"]
        references = ["the cat sat on the mat"]
        
        result = metric.compute(predictions, references)
        
        # No brevity penalty when prediction is longer
        assert result.metadata["brevity_penalty"] == 1.0

    def test_bleu_no_match(self):
        """Test BLEU with no n-gram matches."""
        metric = BLEUScore(n=2)
        
        predictions = ["foo bar"]
        references = ["baz qux"]
        
        result = metric.compute(predictions, references)
        
        assert result.value == 0.0


class TestToxicityScore:
    """Test toxicity detection metric."""

    def test_clean_text(self):
        """Test with clean, non-toxic text."""
        metric = ToxicityScore()
        
        predictions = [
            "This is a nice day.",
            "I appreciate your help.",
            "Thank you very much!",
        ]
        
        result = metric.compute(predictions)
        
        assert result.name == "toxicity"
        assert result.value < 0.2  # Low toxicity
        assert result.metadata["toxic_count"] == 0

    def test_toxic_words(self):
        """Test with known toxic words."""
        metric = ToxicityScore()
        
        predictions = [
            "I hate you so much.",
            "This is awful and terrible.",
            "You should kill that idea.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.3  # Moderate toxicity detected
        assert result.metadata["toxic_count"] > 0

    def test_severe_toxicity(self):
        """Test with severe toxic content."""
        metric = ToxicityScore()
        
        predictions = [
            "fuck this shit",
            "I will kill you",
            "stupid idiot moron",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.6  # High toxicity
        assert result.metadata["max_toxicity"] > 0.8
        assert result.metadata["toxic_rate"] == 1.0  # All toxic

    def test_mixed_toxicity(self):
        """Test with mixed toxic and clean text."""
        metric = ToxicityScore()
        
        predictions = [
            "This is nice.",
            "I hate this.",
            "Great job!",
            "You're stupid.",
        ]
        
        result = metric.compute(predictions)
        
        assert 0.2 < result.value < 0.8  # Mixed toxicity
        assert result.metadata["toxic_rate"] == 0.5  # 2 out of 4


class TestBiasScore:
    """Test bias detection metric."""

    def test_no_bias(self):
        """Test with unbiased text."""
        metric = BiasScore()
        
        predictions = [
            "People come from diverse backgrounds.",
            "Everyone deserves equal treatment.",
            "Individuals have unique capabilities.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.name == "bias"
        assert result.value < 0.2  # Low bias

    def test_gender_bias(self):
        """Test with gender-biased text."""
        metric = BiasScore()
        
        predictions = [
            "Women are too emotional for leadership.",
            "Men can't be good caregivers.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.5  # Significant bias
        assert result.metadata["gender_bias"] > 0.5

    def test_racial_bias(self):
        """Test with racially biased text."""
        metric = BiasScore()
        
        predictions = [
            "Black people are naturally athletic.",
            "Asian people are all good at math.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.5
        assert result.metadata["race_bias"] > 0.5

    def test_religious_bias(self):
        """Test with religious bias."""
        metric = BiasScore()
        
        predictions = [
            "Muslims are all extremists.",
            "Christians are all hypocrites.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.5
        assert result.metadata["religion_bias"] > 0.5

    def test_age_bias(self):
        """Test with age bias."""
        metric = BiasScore()
        
        predictions = [
            "Old people are senile and forgetful.",
            "Young people are lazy and entitled.",
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.5
        assert result.metadata["age_bias"] > 0.5

    def test_multiple_bias_categories(self):
        """Test with bias across multiple categories."""
        metric = BiasScore()
        
        predictions = [
            "Women can't do STEM.",  # Gender
            "Old people are useless.",  # Age
        ]
        
        result = metric.compute(predictions)
        
        assert result.value > 0.5
        # Check multiple categories detected
        biased_categories = sum(1 for k, v in result.metadata.items() 
                               if k.endswith("_bias") and v > 0.3)
        assert biased_categories >= 2


class TestMetricResultDataclass:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        result = MetricResult(
            name="test_metric",
            value=0.85,
            metadata={"key": "value"},
            samples=[{"sample": 1}]
        )
        
        assert result.name == "test_metric"
        assert result.value == 0.85
        assert result.metadata["key"] == "value"
        assert len(result.samples) == 1

    def test_metric_result_no_samples(self):
        """Test MetricResult without samples."""
        result = MetricResult(
            name="test",
            value=0.5,
            metadata={}
        )
        
        assert result.samples is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
