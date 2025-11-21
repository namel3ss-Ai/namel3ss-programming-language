"""
Unit tests for RLHF safety filters.

Tests all 5 filter implementations:
- ToxicityFilter
- PIIFilter
- ProfanityFilter
- BiasFilter
- CompositeFilter
"""

import pytest
from namel3ss.ml.rlhf.safety.filters import (
    ToxicityFilter,
    PIIFilter,
    ProfanityFilter,
    BiasFilter,
    CompositeFilter,
    FilterResult,
    FilterSeverity,
)


class TestToxicityFilter:
    """Test toxicity content filter."""

    def test_clean_text(self):
        """Test with clean, non-toxic text."""
        filter = ToxicityFilter(threshold=0.7)
        
        result = filter.check("This is a wonderful day. Thank you!")
        
        assert result.passed is True
        assert result.filter_name == "toxicity"
        assert result.score < 0.3
        assert len(result.violations) == 0

    def test_mild_toxicity(self):
        """Test with mildly toxic text."""
        filter = ToxicityFilter(threshold=0.7)
        
        result = filter.check("I hate Mondays and this awful weather.")
        
        assert result.score > 0.0  # Some toxicity detected
        # May or may not pass depending on threshold
        if not result.passed:
            assert result.severity in [FilterSeverity.LOW, FilterSeverity.MEDIUM]

    def test_high_toxicity(self):
        """Test with highly toxic content."""
        filter = ToxicityFilter(threshold=0.5)
        
        result = filter.check("I fucking hate you, you stupid idiot!")
        
        assert result.passed is False
        assert result.score > 0.5
        assert result.severity in [FilterSeverity.HIGH, FilterSeverity.CRITICAL]
        assert len(result.violations) > 0

    def test_toxic_patterns(self):
        """Test detection of toxic patterns."""
        filter = ToxicityFilter(threshold=0.5)
        
        result = filter.check("You are so stupid and dumb.")
        
        assert result.score > 0.3  # Pattern detected
        assert len(result.violations) > 0

    def test_threshold_sensitivity(self):
        """Test different threshold settings."""
        text = "This is somewhat negative."
        
        # Strict threshold
        strict_filter = ToxicityFilter(threshold=0.3)
        strict_result = strict_filter.check(text)
        
        # Lenient threshold
        lenient_filter = ToxicityFilter(threshold=0.8)
        lenient_result = lenient_filter.check(text)
        
        # Lenient filter more likely to pass
        if not strict_result.passed:
            assert lenient_result.passed is True

    def test_severity_levels(self):
        """Test severity level assignment."""
        filter = ToxicityFilter()
        
        # Critical severity
        result_critical = filter.check("fuck shit damn kill yourself")
        assert result_critical.severity == FilterSeverity.CRITICAL
        
        # Low severity
        result_low = filter.check("This is a nice day.")
        if result_low.score > 0:
            assert result_low.severity == FilterSeverity.LOW


class TestPIIFilter:
    """Test PII (Personally Identifiable Information) filter."""

    def test_no_pii(self):
        """Test text without PII."""
        filter = PIIFilter()
        
        result = filter.check("I enjoy coding and machine learning.")
        
        assert result.passed is True
        assert result.score == 0.0
        assert len(result.violations) == 0

    def test_email_detection(self):
        """Test email address detection."""
        filter = PIIFilter()
        
        result = filter.check("Contact me at john.doe@example.com for details.")
        
        assert result.passed is False
        assert result.score == 1.0
        assert result.severity == FilterSeverity.CRITICAL
        assert any("email" in v.lower() for v in result.violations)

    def test_phone_number_detection(self):
        """Test phone number detection."""
        filter = PIIFilter()
        
        test_cases = [
            "Call me at 555-123-4567",
            "Phone: (555) 123-4567",
            "My number is 5551234567",
            "+1-555-123-4567",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.passed is False, f"Failed to detect phone in: {text}"
            assert any("phone" in v.lower() for v in result.violations)

    def test_ssn_detection(self):
        """Test Social Security Number detection."""
        filter = PIIFilter()
        
        result = filter.check("My SSN is 123-45-6789")
        
        assert result.passed is False
        assert any("ssn" in v.lower() or "social security" in v.lower() 
                  for v in result.violations)

    def test_credit_card_detection(self):
        """Test credit card number detection."""
        filter = PIIFilter()
        
        test_cases = [
            "Card number: 1234 5678 9012 3456",
            "CC: 1234-5678-9012-3456",
            "Card: 1234567890123456",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.passed is False, f"Failed to detect CC in: {text}"
            assert any("credit card" in v.lower() for v in result.violations)

    def test_ip_address_detection(self):
        """Test IP address detection."""
        filter = PIIFilter()
        
        result = filter.check("Server IP is 192.168.1.1")
        
        assert result.passed is False
        assert any("ip address" in v.lower() for v in result.violations)

    def test_zip_code_detection(self):
        """Test zip code detection."""
        filter = PIIFilter()
        
        test_cases = [
            "ZIP code: 12345",
            "ZIP: 12345-6789",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.passed is False
            assert any("zip" in v.lower() for v in result.violations)

    def test_multiple_pii_types(self):
        """Test text with multiple PII types."""
        filter = PIIFilter()
        
        text = "Email me at john@example.com or call 555-1234. My SSN is 123-45-6789."
        result = filter.check(text)
        
        assert result.passed is False
        assert len(result.violations) >= 3  # Email, phone, SSN

    def test_threshold_ignored_for_pii(self):
        """Test that threshold is ignored (any PII fails)."""
        filter = PIIFilter(threshold=0.9)  # High threshold
        
        result = filter.check("Email: test@example.com")
        
        # Should still fail despite high threshold
        assert result.passed is False
        assert result.score == 1.0


class TestProfanityFilter:
    """Test profanity filter."""

    def test_clean_text(self):
        """Test with clean text."""
        filter = ProfanityFilter()
        
        result = filter.check("This is perfectly clean language.")
        
        assert result.passed is True
        assert result.score == 0.0
        assert len(result.violations) == 0

    def test_common_profanity(self):
        """Test detection of common profane words."""
        filter = ProfanityFilter(threshold=0.5)
        
        result = filter.check("This is fucking ridiculous shit.")
        
        assert result.passed is False
        assert result.score > 0.5
        assert len(result.violations) > 0

    def test_slur_detection(self):
        """Test detection of slurs (higher severity)."""
        filter = ProfanityFilter()
        
        # Note: Using asterisks to avoid actual slurs in test code
        result = filter.check("That person is a total b****.")
        
        # Slurs should trigger critical severity
        if result.score > 0.8:
            assert result.severity == FilterSeverity.CRITICAL

    def test_threshold_filtering(self):
        """Test threshold-based filtering."""
        text = "damn it"
        
        # Strict threshold
        strict_filter = ProfanityFilter(threshold=0.1)
        strict_result = strict_filter.check(text)
        
        # Lenient threshold
        lenient_filter = ProfanityFilter(threshold=0.8)
        lenient_result = lenient_filter.check(text)
        
        # Strict filter more likely to fail
        if strict_result.passed is False:
            assert lenient_result.passed is True

    def test_custom_word_list(self):
        """Test custom profane word list."""
        custom_words = ["badword", "naughty"]
        filter = ProfanityFilter(custom_words=custom_words)
        
        result = filter.check("This contains badword and naughty.")
        
        assert result.passed is False
        assert result.score > 0.0

    def test_case_insensitivity(self):
        """Test case-insensitive detection."""
        filter = ProfanityFilter()
        
        result1 = filter.check("FUCK")
        result2 = filter.check("fuck")
        result3 = filter.check("FuCk")
        
        assert result1.score == result2.score == result3.score

    def test_partial_word_matching(self):
        """Test that filter doesn't match partial words incorrectly."""
        filter = ProfanityFilter()
        
        # "class" contains "ass" but shouldn't be flagged
        result = filter.check("This is a good class.")
        
        # Should have very low or zero score
        assert result.score < 0.3


class TestBiasFilter:
    """Test bias detection filter."""

    def test_no_bias(self):
        """Test with unbiased text."""
        filter = BiasFilter()
        
        result = filter.check("People of all backgrounds contribute to society.")
        
        assert result.passed is True
        assert result.score < 0.3
        assert len(result.violations) == 0

    def test_gender_bias(self):
        """Test gender bias detection."""
        filter = BiasFilter(threshold=0.5)
        
        test_cases = [
            "Women are too emotional to be leaders.",
            "Men can't be nurturing parents.",
            "Women belong in the kitchen.",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.score > 0.3, f"Failed to detect bias in: {text}"
            assert any("gender" in v.lower() for v in result.violations)

    def test_racial_bias(self):
        """Test racial bias detection."""
        filter = BiasFilter(threshold=0.5)
        
        test_cases = [
            "Black people are naturally good at sports.",
            "Asian people are all good at math.",
            "White people are all privileged.",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.score > 0.3
            assert any("race" in v.lower() or "racial" in v.lower() 
                      for v in result.violations)

    def test_religious_bias(self):
        """Test religious bias detection."""
        filter = BiasFilter(threshold=0.5)
        
        test_cases = [
            "Muslims are all terrorists.",
            "Christians are all extremists.",
            "Jews are all greedy.",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.score > 0.3
            assert any("religion" in v.lower() for v in result.violations)

    def test_age_bias(self):
        """Test age bias detection."""
        filter = BiasFilter(threshold=0.5)
        
        test_cases = [
            "Old people are senile and useless.",
            "Young people are lazy and entitled.",
            "Millennials are killing industries.",
        ]
        
        for text in test_cases:
            result = filter.check(text)
            assert result.score > 0.3
            assert any("age" in v.lower() for v in result.violations)

    def test_multiple_bias_types(self):
        """Test detection of multiple bias types."""
        filter = BiasFilter()
        
        text = "Women can't do math. Old people can't use technology."
        result = filter.check(text)
        
        assert result.passed is False
        assert result.score > 0.5
        # Should detect both gender and age bias
        assert len([v for v in result.violations if "gender" in v.lower()]) > 0
        assert len([v for v in result.violations if "age" in v.lower()]) > 0

    def test_threshold_filtering(self):
        """Test threshold-based filtering."""
        text = "Some people think women are emotional."
        
        # Strict threshold
        strict_filter = BiasFilter(threshold=0.3)
        strict_result = strict_filter.check(text)
        
        # Lenient threshold
        lenient_filter = BiasFilter(threshold=0.8)
        lenient_result = lenient_filter.check(text)
        
        # Different pass/fail based on threshold
        if not strict_result.passed:
            assert lenient_result.passed is True


class TestCompositeFilter:
    """Test composite filter combining multiple filters."""

    def test_all_filters_pass(self):
        """Test when all filters pass."""
        filters = [
            ToxicityFilter(threshold=0.7),
            PIIFilter(),
            ProfanityFilter(threshold=0.7),
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("This is clean, appropriate text.")
        
        assert result.passed is True
        assert result.filter_name == "composite"
        assert len(result.violations) == 0

    def test_one_filter_fails_require_all(self):
        """Test when one filter fails with require_all=True."""
        filters = [
            ToxicityFilter(threshold=0.5),
            PIIFilter(),
            ProfanityFilter(threshold=0.5),
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        # Contains PII but otherwise clean
        result = composite.check("Contact me at test@example.com")
        
        assert result.passed is False
        assert len(result.violations) > 0

    def test_require_any_logic(self):
        """Test require_all=False (any filter passing is ok)."""
        filters = [
            ToxicityFilter(threshold=0.3),  # Very strict
            ProfanityFilter(threshold=0.9),  # Very lenient
        ]
        composite = CompositeFilter(filters, require_all=False)
        
        # Text with mild profanity
        result = composite.check("This is damn good.")
        
        # Should pass if lenient filter passes
        assert result.passed is True

    def test_all_filters_fail_require_all_false(self):
        """Test when all filters fail with require_all=False."""
        filters = [
            ToxicityFilter(threshold=0.3),
            ProfanityFilter(threshold=0.3),
        ]
        composite = CompositeFilter(filters, require_all=False)
        
        result = composite.check("This fucking sucks, I hate it.")
        
        # All filters should fail
        assert result.passed is False

    def test_violation_aggregation(self):
        """Test aggregation of violations from multiple filters."""
        filters = [
            ToxicityFilter(threshold=0.5),
            ProfanityFilter(threshold=0.5),
            PIIFilter(),
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("Fuck this shit. Email me at test@example.com")
        
        # Should have violations from multiple filters
        assert len(result.violations) >= 2

    def test_max_severity_selection(self):
        """Test that max severity is selected."""
        filters = [
            ToxicityFilter(threshold=0.5),  # Might give MEDIUM
            PIIFilter(),  # Gives CRITICAL
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("My email is test@example.com")
        
        # Should take CRITICAL from PIIFilter
        assert result.severity == FilterSeverity.CRITICAL

    def test_average_score_calculation(self):
        """Test score averaging across filters."""
        filters = [
            ToxicityFilter(threshold=0.9),
            ProfanityFilter(threshold=0.9),
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("This is clean text.")
        
        # Average of low scores should be low
        assert result.score < 0.2

    def test_individual_results_in_metadata(self):
        """Test individual filter results in metadata."""
        filters = [
            ToxicityFilter(threshold=0.7),
            PIIFilter(),
        ]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("Test text")
        
        assert "individual_results" in result.metadata
        assert len(result.metadata["individual_results"]) == 2

    def test_empty_filter_list(self):
        """Test with empty filter list."""
        composite = CompositeFilter([], require_all=True)
        
        result = composite.check("Any text")
        
        # Should pass with no filters
        assert result.passed is True
        assert result.score == 0.0

    def test_single_filter(self):
        """Test with single filter (edge case)."""
        filters = [ToxicityFilter(threshold=0.5)]
        composite = CompositeFilter(filters, require_all=True)
        
        result = composite.check("Clean text")
        
        # Should behave like single filter
        assert result.passed is True


class TestFilterResultDataclass:
    """Test FilterResult dataclass."""

    def test_filter_result_creation(self):
        """Test creating FilterResult."""
        result = FilterResult(
            passed=False,
            filter_name="test_filter",
            severity=FilterSeverity.HIGH,
            score=0.75,
            violations=["violation 1", "violation 2"],
            metadata={"key": "value"}
        )
        
        assert result.passed is False
        assert result.filter_name == "test_filter"
        assert result.severity == FilterSeverity.HIGH
        assert result.score == 0.75
        assert len(result.violations) == 2
        assert result.metadata["key"] == "value"

    def test_filter_result_no_severity(self):
        """Test FilterResult without severity."""
        result = FilterResult(
            passed=True,
            filter_name="test",
            severity=None,
            score=0.1,
            violations=[],
            metadata={}
        )
        
        assert result.severity is None


class TestFilterSeverityEnum:
    """Test FilterSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert FilterSeverity.LOW
        assert FilterSeverity.MEDIUM
        assert FilterSeverity.HIGH
        assert FilterSeverity.CRITICAL

    def test_severity_values(self):
        """Test severity enum values."""
        assert FilterSeverity.LOW.value == "low"
        assert FilterSeverity.MEDIUM.value == "medium"
        assert FilterSeverity.HIGH.value == "high"
        assert FilterSeverity.CRITICAL.value == "critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
