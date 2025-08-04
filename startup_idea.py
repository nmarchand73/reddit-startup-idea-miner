#!/usr/bin/env python3
"""
StartupIdea dataclass for representing and analyzing startup ideas
"""

import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any
from constants import SUCCESS_INDICATORS, PROBLEM_INDICATORS

@dataclass
class StartupIdea:
    """Represents a startup idea extracted from Reddit"""
    title: str
    subreddit: str
    score: int
    url: str
    content: str = ""  # Add missing content field
    author: str = ""  # Add missing author field
    num_comments: int = 0  # Add missing num_comments field
    created_utc: float = 0.0  # Add missing created_utc field
    pain_score: float = 0.0
    monetization_model: str = "Unknown"
    market_size: str = "Unknown"
    validation_signals: List[str] = None  # Changed to List[str]
    engagement_velocity: float = 0.0
    comments: int = 0
    content_preview: str = ""
    is_problem_statement: bool = False
    is_success_story: bool = False
    specificity_score: float = 0.0
    frequency_score: float = 0.0
    diy_score: float = 0.0
    money_score: float = 0.0
    endorsement_score: float = 0.0
    id: int = None  # Unique identifier for ranking
    
    def __post_init__(self):
        if self.validation_signals is None:
            self.validation_signals = []
        # Ensure content is not None
        if self.content is None:
            self.content = ""
        # Generate content hash for deduplication
        self.content_hash = self._generate_hash()
        # Classify content type
        self._classify_content_type()
    
    def _generate_hash(self) -> str:
        """Generate hash for deduplication"""
        content = f"{self.title} {self.content}".lower()
        return hashlib.md5(content.encode()).hexdigest()
    
    def _classify_content_type(self):
        """Classify if content is a problem statement or success story"""
        text = f"{self.title} {self.content}".lower()
        
        success_count = sum(1 for indicator in SUCCESS_INDICATORS if indicator in text)
        problem_count = sum(1 for indicator in PROBLEM_INDICATORS if indicator in text)
        
        self.is_success_story = success_count > problem_count
        self.is_problem_statement = problem_count > success_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the startup idea to a dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'subreddit': self.subreddit,
            'pain_score': self.pain_score,
            'monetization_model': self.monetization_model,
            'market_size': self.market_size,
            'validation_signals': self.validation_signals,
            'engagement_velocity': self.engagement_velocity,
            'score': self.score,
            'comments': self.comments,
            'url': self.url,
            'content_preview': self.content_preview,
            'is_problem_statement': self.is_problem_statement,
            'is_success_story': self.is_success_story,
            'specificity_score': self.specificity_score,
            'frequency_score': self.frequency_score,
            'diy_score': self.diy_score,
            'money_score': self.money_score,
            'endorsement_score': self.endorsement_score
        }
    
    def get_content_preview(self, max_length: int = 200) -> str:
        """Get a preview of the content"""
        if not self.content:
            return ""
        return self.content[:max_length] + "..." if len(self.content) > max_length else self.content
    
    def is_high_potential(self) -> bool:
        """Check if this idea has high potential based on pain score"""
        return self.pain_score >= 15
    
    def is_exceptional_potential(self) -> bool:
        """Check if this idea has exceptional potential"""
        return self.pain_score >= 20
    
    def get_validation_summary(self) -> str:
        """Get a summary of validation metrics"""
        return f"Pain: {self.pain_score:.1f}, Engagement: {self.engagement_velocity:.3f}, Model: {self.monetization_model}" 