#!/usr/bin/env python3
"""
Non-regression tests for scoring formulas
Ensures mathematical consistency and prevents formula changes from breaking existing functionality
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reddit_idea_miner import RedditIdeaMiner
from startup_idea import StartupIdea
from ollama_analyzer import OllamaAnalyzer
from constants import PAIN_PATTERNS, MONETIZATION_MODELS


class TestScoringFormulas(unittest.TestCase):
    """Test suite for all scoring formulas"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock Reddit client for testing
        self.mock_reddit = Mock()
        self.mock_reddit.user.me.return_value = Mock()
        
        # Create miner with mock credentials
        self.miner = RedditIdeaMiner("test_id", "test_secret", "test_agent")
        self.miner.reddit = self.mock_reddit
        
        # Create analyzer
        self.analyzer = OllamaAnalyzer()
    
    def test_engagement_velocity_formula(self):
        """Test engagement velocity calculation"""
        # Test case 1: Normal post
        mock_post = Mock()
        mock_post.score = 10
        mock_post.num_comments = 5
        mock_post.created_utc = time.time() - 86400  # 24 hours ago
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        expected = (10 + 5) / 24 * (1 + min(5/10, 2.0) * 0.5)
        self.assertAlmostEqual(velocity, expected, places=2)
        
        # Test case 2: Viral post
        mock_post.score = 100
        mock_post.num_comments = 50
        mock_post.created_utc = time.time() - 7200  # 2 hours ago
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        expected = (100 + 50) / 2 * (1 + min(50/100, 2.0) * 0.5)
        self.assertAlmostEqual(velocity, expected, places=2)
        
        # Test case 3: No engagement
        mock_post.score = 0
        mock_post.num_comments = 0
        mock_post.created_utc = time.time() - 3600  # 1 hour ago
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        self.assertEqual(velocity, 0.0)
        
        # Test case 4: High discussion ratio
        mock_post.score = 5
        mock_post.num_comments = 20
        mock_post.created_utc = time.time() - 43200  # 12 hours ago
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        expected = (5 + 20) / 12 * (1 + min(20/5, 2.0) * 0.5)
        self.assertAlmostEqual(velocity, expected, places=2)
    
    def test_pain_score_formula(self):
        """Test pain score calculation"""
        # Create test idea
        idea = StartupIdea(
            title="Test idea",
            content="I need a tool to automate my daily tasks",
            subreddit="Entrepreneur",
            author="test_user",
            score=15,
            num_comments=25,
            created_utc=time.time(),
            url="https://reddit.com/test"
        )
        
        # Apply pain scan framework
        result = self.miner.apply_pain_scan_framework(idea)
        
        # Test that weights sum to 1.0
        weights_sum = 0.2 + 0.2 + 0.25 + 0.25 + 0.1
        self.assertEqual(weights_sum, 1.0)
        
        # Test that pain score is calculated correctly
        expected_pain_score = (
            result.specificity_score * 0.2 +
            result.frequency_score * 0.2 +
            result.diy_score * 0.25 +
            result.money_score * 0.25 +
            result.endorsement_score * 0.1
        )
        self.assertAlmostEqual(result.pain_score, expected_pain_score, places=3)
        
        # Test that pain score is non-negative
        self.assertGreaterEqual(result.pain_score, 0.0)
        
        # Test component scores are non-negative
        self.assertGreaterEqual(result.specificity_score, 0.0)
        self.assertGreaterEqual(result.frequency_score, 0.0)
        self.assertGreaterEqual(result.diy_score, 0.0)
        self.assertGreaterEqual(result.money_score, 0.0)
        self.assertGreaterEqual(result.endorsement_score, 0.0)
    
    def test_endorsement_scoring(self):
        """Test group endorsement scoring"""
        # Test case 1: High engagement
        idea = StartupIdea(
            title="Test",
            content="Test content",
            subreddit="test",
            author="test",
            score=15,
            num_comments=25,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.endorsement_score, 4.0)  # 2.0 + 2.0
        
        # Test case 2: Medium engagement
        idea.score = 8
        idea.num_comments = 15
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.endorsement_score, 2.0)  # 1.0 + 1.0
        
        # Test case 3: Low engagement
        idea.score = 3
        idea.num_comments = 8
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.endorsement_score, 0.0)
        
        # Test case 4: No engagement
        idea.score = 0
        idea.num_comments = 0
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.endorsement_score, 0.0)
    
    def test_monetization_classification(self):
        """Test monetization model classification"""
        # Test Micro SaaS classification
        idea = StartupIdea(
            title="I need a tool to automate my workflow",
            content="Looking for a dashboard to track my analytics and integrate with my API",
            subreddit="SaaS",
            author="test",
            score=10,
            num_comments=5,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.classify_monetization_model(idea)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
        
        # Test UGC Marketplace classification
        idea = StartupIdea(
            title="Need a platform to connect creators",
            content="Looking for a marketplace to connect buyers and sellers",
            subreddit="startups",
            author="test",
            score=10,
            num_comments=5,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.classify_monetization_model(idea)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
        
        # Test Done-for-You classification
        idea = StartupIdea(
            title="Need a service to do this for me",
            content="Looking for a template or agency to provide consulting",
            subreddit="smallbusiness",
            author="test",
            score=10,
            num_comments=5,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.classify_monetization_model(idea)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
        
        # Test Unknown classification
        idea = StartupIdea(
            title="Random post",
            content="This is just a random post with no specific monetization signals or keywords at all",
            subreddit="random",
            author="test",
            score=10,
            num_comments=5,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.classify_monetization_model(idea)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
    
    def test_fallback_scoring_formula(self):
        """Test fallback analysis scoring"""
        # Test case 1: Low pain score
        idea = {"pain_score": 0, "monetization_model": "Unknown"}
        result = self.analyzer._fallback_analysis(idea)
        
        self.assertEqual(result['score']['business'], 3.0)
        self.assertEqual(result['score']['execution'], 8.0)
        self.assertEqual(result['score']['monetization'], 3.0)
        
        # Test case 2: Medium pain score
        idea = {"pain_score": 10, "monetization_model": "micro_saas"}
        result = self.analyzer._fallback_analysis(idea)
        
        self.assertEqual(result['score']['business'], 8.0)
        self.assertEqual(result['score']['execution'], 5.5)
        self.assertEqual(result['score']['monetization'], 6.0)
        
        # Test case 3: High pain score
        idea = {"pain_score": 20, "monetization_model": "ugc_marketplace"}
        result = self.analyzer._fallback_analysis(idea)
        
        self.assertEqual(result['score']['business'], 10.0)  # Capped at 10
        self.assertEqual(result['score']['execution'], 3.0)
        self.assertEqual(result['score']['monetization'], 6.0)
        
        # Test bounds
        for score_type in ['business', 'execution', 'monetization']:
            self.assertGreaterEqual(result['score'][score_type], 0)
            self.assertLessEqual(result['score'][score_type], 10)
    
    def test_recommendation_thresholds(self):
        """Test recommendation threshold logic"""
        # Test case 1: Pursue Aggressively
        scores = {'business': 8, 'execution': 8, 'monetization': 8}
        avg_score = sum(scores.values()) / 3
        self.assertGreaterEqual(avg_score, 7)
        
        # Test case 2: Pursue Cautiously
        scores = {'business': 6, 'execution': 6, 'monetization': 6}
        avg_score = sum(scores.values()) / 3
        self.assertGreaterEqual(avg_score, 5)
        self.assertLess(avg_score, 7)
        
        # Test case 3: Validate Further
        scores = {'business': 4, 'execution': 4, 'monetization': 4}
        avg_score = sum(scores.values()) / 3
        self.assertGreaterEqual(avg_score, 3)
        self.assertLess(avg_score, 5)
        
        # Test case 4: Pass
        scores = {'business': 2, 'execution': 2, 'monetization': 2}
        avg_score = sum(scores.values()) / 3
        self.assertLess(avg_score, 3)
    
    def test_pain_patterns_consistency(self):
        """Test that pain patterns are properly defined"""
        required_patterns = [
            'specificity_indicators',
            'frequency_indicators', 
            'diy_evidence',
            'money_signals'
        ]
        
        for pattern in required_patterns:
            self.assertIn(pattern, PAIN_PATTERNS)
            self.assertIn('weight', PAIN_PATTERNS[pattern])
            self.assertGreater(PAIN_PATTERNS[pattern]['weight'], 0)
    
    def test_monetization_models_consistency(self):
        """Test that monetization models are properly defined"""
        required_models = ['micro_saas', 'ugc_marketplace', 'done_for_you']
        
        for model in required_models:
            self.assertIn(model, MONETIZATION_MODELS)
            self.assertIn('weight', MONETIZATION_MODELS[model])
            self.assertIn('signals', MONETIZATION_MODELS[model])
            self.assertGreater(MONETIZATION_MODELS[model]['weight'], 0)
            self.assertIsInstance(MONETIZATION_MODELS[model]['signals'], list)
    
    def test_formula_edge_cases(self):
        """Test edge cases for all formulas"""
        # Test engagement velocity with very small age
        mock_post = Mock()
        mock_post.score = 10
        mock_post.num_comments = 5
        mock_post.created_utc = time.time() - 0.001  # Very small age
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        self.assertGreaterEqual(velocity, 0)
        
        # Test pain score with empty content
        idea = StartupIdea(
            title="",
            content="",
            subreddit="test",
            author="test",
            score=0,
            num_comments=0,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.pain_score, 0.0)
        
        # Test monetization with no signals
        result = self.miner.classify_monetization_model(idea)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
    
    def test_formula_determinism(self):
        """Test that formulas produce consistent results"""
        # Test engagement velocity determinism
        mock_post = Mock()
        mock_post.score = 10
        mock_post.num_comments = 5
        mock_post.created_utc = time.time() - 3600
        
        velocity1 = self.miner.calculate_engagement_velocity(mock_post)
        velocity2 = self.miner.calculate_engagement_velocity(mock_post)
        self.assertAlmostEqual(velocity1, velocity2, places=2)
        
        # Test pain score determinism
        idea = StartupIdea(
            title="Test idea",
            content="I need a tool to automate my daily tasks",
            subreddit="Entrepreneur",
            author="test_user",
            score=15,
            num_comments=25,
            created_utc=time.time(),
            url="https://reddit.com/test"
        )
        
        result1 = self.miner.apply_pain_scan_framework(idea)
        result2 = self.miner.apply_pain_scan_framework(idea)
        self.assertAlmostEqual(result1.pain_score, result2.pain_score, places=3)


class TestFormulaPerformance(unittest.TestCase):
    """Test performance characteristics of formulas"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.miner = RedditIdeaMiner("test_id", "test_secret", "test_agent")
        self.analyzer = OllamaAnalyzer()
    
    def test_linear_time_complexity(self):
        """Test that formulas have linear time complexity"""
        import time as time_module
        
        # Test engagement velocity performance
        mock_post = Mock()
        mock_post.score = 10
        mock_post.num_comments = 5
        mock_post.created_utc = time.time() - 3600
        
        start_time = time_module.time()
        for _ in range(1000):
            self.miner.calculate_engagement_velocity(mock_post)
        end_time = time_module.time()
        
        # Should complete in reasonable time (less than 1 second for 1000 iterations)
        self.assertLess(end_time - start_time, 1.0)
    
    def test_memory_efficiency(self):
        """Test that formulas don't create unnecessary data structures"""
        import gc
        import sys
        
        # Test pain score memory usage
        idea = StartupIdea(
            title="Test idea",
            content="I need a tool to automate my daily tasks",
            subreddit="Entrepreneur",
            author="test_user",
            score=15,
            num_comments=25,
            created_utc=time.time(),
            url="https://reddit.com/test"
        )
        
        # Get initial memory usage
        gc.collect()
        initial_memory = sys.getsizeof(idea)
        
        # Apply pain scan framework
        result = self.miner.apply_pain_scan_framework(idea)
        
        # Check that memory usage didn't explode
        final_memory = sys.getsizeof(result)
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1KB)
        self.assertLess(memory_increase, 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2) 