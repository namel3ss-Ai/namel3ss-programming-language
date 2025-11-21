"""
Safety filters for content moderation in RLHF training and inference.

This module provides filters for detecting and blocking harmful content:
- Toxicity: Offensive, hateful, or abusive language
- PII: Personal identifiable information (emails, phones, SSNs)
- Profanity: Explicit language and slurs
- Bias: Demographic biases and stereotypes

Filters can be used during training to filter datasets or during
inference to block unsafe model outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re
from enum import Enum


class FilterSeverity(Enum):
    """Severity levels for filter violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    """Result of applying a safety filter."""
    passed: bool  # True if content is safe
    filter_name: str
    severity: Optional[FilterSeverity] = None
    score: float = 0.0  # 0-1, higher means more problematic
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyFilter(ABC):
    """
    Base class for safety filters.
    
    Safety filters analyze text and determine if it contains
    harmful, inappropriate, or sensitive content.
    """
    
    def __init__(self, name: str, threshold: float = 0.5):
        """
        Initialize filter.
        
        Args:
            name: Filter name
            threshold: Score threshold for blocking (0-1)
        """
        self.name = name
        self.threshold = threshold
    
    @abstractmethod
    def check(self, text: str) -> FilterResult:
        """
        Check if text passes the safety filter.
        
        Args:
            text: Text to analyze
        
        Returns:
            FilterResult with pass/fail and details
        """
        pass
    
    def batch_check(self, texts: List[str]) -> List[FilterResult]:
        """
        Check multiple texts.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of FilterResults
        """
        return [self.check(text) for text in texts]


class ToxicityFilter(SafetyFilter):
    """
    Toxicity filter using rule-based and ML approaches.
    
    Detects toxic, hateful, threatening, or abusive content.
    Can use Perspective API for production deployments.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        use_perspective_api: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialize toxicity filter.
        
        Args:
            threshold: Toxicity threshold (0-1)
            use_perspective_api: Use Google Perspective API
            api_key: Perspective API key
        """
        super().__init__("toxicity", threshold)
        self.use_perspective_api = use_perspective_api
        self.api_key = api_key
        
        # Rule-based toxic words list
        self.toxic_words = {
            'hate', 'kill', 'murder', 'death', 'destroy',
            'stupid', 'idiot', 'moron', 'dumb', 'retard',
            'fuck', 'shit', 'damn', 'hell', 'bitch',
            'racist', 'sexist', 'bigot', 'nazi'
        }
        
        # Toxic patterns
        self.toxic_patterns = [
            r'you\s+(are|r)\s+(stupid|dumb|idiot)',
            r'i\s+hate\s+you',
            r'go\s+to\s+hell',
            r'kill\s+yourself',
        ]
    
    def check(self, text: str) -> FilterResult:
        """Check text for toxicity."""
        if self.use_perspective_api and self.api_key:
            return self._check_perspective_api(text)
        else:
            return self._check_rule_based(text)
    
    def _check_rule_based(self, text: str) -> FilterResult:
        """Rule-based toxicity detection."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Check for toxic words
        toxic_matches = words & self.toxic_words
        
        # Check for toxic patterns
        pattern_matches = []
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                pattern_matches.append(pattern)
        
        # Calculate toxicity score
        word_score = len(toxic_matches) / len(words) if words else 0.0
        pattern_score = len(pattern_matches) * 0.3
        score = min(word_score + pattern_score, 1.0)
        
        # Determine severity
        if score >= 0.8:
            severity = FilterSeverity.CRITICAL
        elif score >= 0.6:
            severity = FilterSeverity.HIGH
        elif score >= 0.3:
            severity = FilterSeverity.MEDIUM
        else:
            severity = FilterSeverity.LOW
        
        violations = list(toxic_matches) + pattern_matches
        passed = score < self.threshold
        
        return FilterResult(
            passed=passed,
            filter_name=self.name,
            severity=severity,
            score=score,
            violations=violations,
            metadata={
                "toxic_words": list(toxic_matches),
                "toxic_patterns": pattern_matches,
                "word_count": len(words)
            }
        )
    
    def _check_perspective_api(self, text: str) -> FilterResult:
        """
        Check toxicity using Perspective API.
        
        Note: This is a placeholder. Production implementation should
        use the actual Perspective API client.
        """
        # Placeholder - would make API call here
        return FilterResult(
            passed=True,
            filter_name=self.name,
            score=0.0,
            metadata={"error": "Perspective API not implemented"}
        )


class PIIFilter(SafetyFilter):
    """
    Personally Identifiable Information (PII) filter.
    
    Detects and blocks content containing:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize PII filter.
        
        Args:
            threshold: PII threshold (0-1, usually 0 for strict filtering)
        """
        super().__init__("pii", threshold)
        
        # PII regex patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "zip_code": r'\b\d{5}(-\d{4})?\b',
        }
    
    def check(self, text: str) -> FilterResult:
        """Check text for PII."""
        violations = []
        pii_types = []
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.extend([f"{pii_type}: {match}" for match in matches])
                pii_types.append(pii_type)
        
        # Score is binary - either PII detected or not
        score = 1.0 if violations else 0.0
        passed = score <= self.threshold
        
        severity = FilterSeverity.CRITICAL if not passed else None
        
        return FilterResult(
            passed=passed,
            filter_name=self.name,
            severity=severity,
            score=score,
            violations=violations,
            metadata={
                "pii_types": pii_types,
                "pii_count": len(violations)
            }
        )


class ProfanityFilter(SafetyFilter):
    """
    Profanity and explicit content filter.
    
    Detects profane language, slurs, and explicit sexual content.
    """
    
    def __init__(self, threshold: float = 0.5, custom_words: Optional[Set[str]] = None):
        """
        Initialize profanity filter.
        
        Args:
            threshold: Profanity threshold (0-1)
            custom_words: Additional profane words to block
        """
        super().__init__("profanity", threshold)
        
        # Profane words list (partial - production would use comprehensive list)
        self.profane_words = {
            'fuck', 'fucking', 'fucked', 'fucker',
            'shit', 'shit', 'shitty', 'bullshit',
            'ass', 'asshole', 'bastard', 'bitch',
            'damn', 'hell', 'crap', 'piss',
        }
        
        if custom_words:
            self.profane_words.update(custom_words)
        
        # Slurs and hate speech (placeholder - production needs comprehensive list)
        self.slurs = {
            # Placeholder - actual implementation would have comprehensive list
            'slur1', 'slur2',  # Replaced with actual slurs in production
        }
    
    def check(self, text: str) -> FilterResult:
        """Check text for profanity."""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for profane words
        profane_matches = []
        for word in words:
            if word in self.profane_words or word in self.slurs:
                profane_matches.append(word)
        
        # Calculate profanity score
        score = len(profane_matches) / len(words) if words else 0.0
        passed = score < self.threshold
        
        # Determine severity
        has_slurs = any(word in self.slurs for word in profane_matches)
        if has_slurs:
            severity = FilterSeverity.CRITICAL
        elif score >= 0.5:
            severity = FilterSeverity.HIGH
        elif score >= 0.2:
            severity = FilterSeverity.MEDIUM
        else:
            severity = FilterSeverity.LOW if profane_matches else None
        
        return FilterResult(
            passed=passed,
            filter_name=self.name,
            severity=severity,
            score=score,
            violations=profane_matches,
            metadata={
                "profane_word_count": len(profane_matches),
                "has_slurs": has_slurs,
                "total_words": len(words)
            }
        )


class BiasFilter(SafetyFilter):
    """
    Demographic bias filter.
    
    Detects potential biases related to:
    - Gender
    - Race/ethnicity
    - Religion
    - Age
    - Sexual orientation
    - Disability
    """
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize bias filter.
        
        Args:
            threshold: Bias threshold (0-1)
        """
        super().__init__("bias", threshold)
        
        # Bias indicators by category
        self.bias_indicators = {
            "gender": {
                'patterns': [
                    r'women\s+(are|can\'t|shouldn\'t)',
                    r'men\s+(are|should|must)',
                    r'(girls|boys)\s+are\s+(better|worse)',
                ],
                'stereotypes': ['emotional', 'aggressive', 'nurturing', 'strong']
            },
            "race": {
                'patterns': [
                    r'(black|white|asian|hispanic)\s+people\s+are',
                ],
                'stereotypes': ['lazy', 'smart', 'criminal', 'good at']
            },
            "religion": {
                'patterns': [
                    r'(muslims|christians|jews|hindus)\s+are',
                ],
                'stereotypes': ['terrorist', 'extremist', 'greedy']
            },
            "age": {
                'patterns': [
                    r'(young|old)\s+people\s+(are|can\'t)',
                ],
                'stereotypes': ['lazy', 'senile', 'entitled', 'out of touch']
            }
        }
    
    def check(self, text: str) -> FilterResult:
        """Check text for demographic bias."""
        text_lower = text.lower()
        
        violations = []
        bias_types = []
        total_score = 0.0
        
        for bias_type, indicators in self.bias_indicators.items():
            type_score = 0.0
            
            # Check patterns
            for pattern in indicators['patterns']:
                if re.search(pattern, text_lower):
                    violations.append(f"{bias_type}: {pattern}")
                    type_score += 0.5
            
            # Check stereotypes
            for stereotype in indicators['stereotypes']:
                if stereotype in text_lower:
                    violations.append(f"{bias_type} stereotype: {stereotype}")
                    type_score += 0.3
            
            if type_score > 0:
                bias_types.append(bias_type)
                total_score += type_score
        
        # Normalize score
        score = min(total_score, 1.0)
        passed = score < self.threshold
        
        # Determine severity
        if score >= 0.8:
            severity = FilterSeverity.HIGH
        elif score >= 0.5:
            severity = FilterSeverity.MEDIUM
        else:
            severity = FilterSeverity.LOW if violations else None
        
        return FilterResult(
            passed=passed,
            filter_name=self.name,
            severity=severity,
            score=score,
            violations=violations,
            metadata={
                "bias_types": bias_types,
                "violation_count": len(violations)
            }
        )


class CompositeFilter(SafetyFilter):
    """
    Composite filter combining multiple safety filters.
    
    Runs multiple filters and aggregates results.
    Content passes only if it passes all filters.
    """
    
    def __init__(
        self,
        filters: List[SafetyFilter],
        name: str = "composite",
        require_all: bool = True
    ):
        """
        Initialize composite filter.
        
        Args:
            filters: List of filters to apply
            name: Filter name
            require_all: If True, all filters must pass. If False, any filter passing is sufficient.
        """
        super().__init__(name, threshold=0.0)
        self.filters = filters
        self.require_all = require_all
    
    def check(self, text: str) -> FilterResult:
        """Check text against all filters."""
        results = [f.check(text) for f in self.filters]
        
        if self.require_all:
            passed = all(r.passed for r in results)
        else:
            passed = any(r.passed for r in results)
        
        # Aggregate violations and scores
        all_violations = []
        max_severity = None
        total_score = 0.0
        
        for result in results:
            all_violations.extend(result.violations)
            total_score += result.score
            
            if result.severity:
                if not max_severity or self._severity_level(result.severity) > self._severity_level(max_severity):
                    max_severity = result.severity
        
        avg_score = total_score / len(results) if results else 0.0
        
        return FilterResult(
            passed=passed,
            filter_name=self.name,
            severity=max_severity,
            score=avg_score,
            violations=all_violations,
            metadata={
                "filter_results": [
                    {
                        "filter": r.filter_name,
                        "passed": r.passed,
                        "score": r.score,
                        "severity": r.severity.value if r.severity else None
                    }
                    for r in results
                ],
                "filters_passed": sum(1 for r in results if r.passed),
                "filters_failed": sum(1 for r in results if not r.passed)
            }
        )
    
    def _severity_level(self, severity: FilterSeverity) -> int:
        """Convert severity to numeric level for comparison."""
        levels = {
            FilterSeverity.LOW: 1,
            FilterSeverity.MEDIUM: 2,
            FilterSeverity.HIGH: 3,
            FilterSeverity.CRITICAL: 4
        }
        return levels.get(severity, 0)


__all__ = [
    "SafetyFilter",
    "FilterResult",
    "FilterSeverity",
    "ToxicityFilter",
    "PIIFilter",
    "ProfanityFilter",
    "BiasFilter",
    "CompositeFilter",
]
