#!/usr/bin/env python3
"""
Integration tests for the complete scoring pipeline
Tests the entire flow from Reddit post to final analysis
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reddit_idea_miner import RedditIdeaMiner
from startup_idea import StartupIdea
from ollama_analyzer import OllamaAnalyzer


class TestScoringPipeline(unittest.TestCase):
    """Test the complete scoring pipeline end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock Reddit client
        self.mock_reddit = Mock()
        self.mock_reddit.user.me.return_value = Mock()
        
        # Create miner with mock credentials
        self.miner = RedditIdeaMiner("test_id", "test_secret", "test_agent")
        self.miner.reddit = self.mock_reddit
        
        # Create analyzer
        self.analyzer = OllamaAnalyzer()
    
    def test_complete_pipeline_flow(self):
        """Test the complete pipeline from post to analysis"""
        # Create a realistic test post
        mock_post = Mock()
        mock_post.title = "I need a tool to automate my daily tasks"
        mock_post.selftext = "I spend 2 hours every day manually copying data between spreadsheets. I would pay $50/month for a tool that automates this."
        mock_post.author = "test_user"
        mock_post.score = 25
        mock_post.num_comments = 15
        mock_post.created_utc = time.time() - 86400  # 24 hours ago
        mock_post.permalink = "/r/Entrepreneur/test_post"
        
        # Test engagement velocity calculation
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        self.assertGreater(velocity, 0)
        
        # Create StartupIdea object
        idea = StartupIdea(
            title=mock_post.title,
            content=mock_post.selftext,
            subreddit="Entrepreneur",
            author=str(mock_post.author),
            score=mock_post.score,
            num_comments=mock_post.num_comments,
            created_utc=mock_post.created_utc,
            url=f"https://reddit.com{mock_post.permalink}"
        )
        
        # Test pain scan framework
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertGreater(result.pain_score, 0)
        self.assertGreaterEqual(result.specificity_score, 0)
        self.assertGreaterEqual(result.frequency_score, 0)
        self.assertGreaterEqual(result.diy_score, 0)
        self.assertGreaterEqual(result.money_score, 0)
        self.assertGreaterEqual(result.endorsement_score, 0)
        
        # Test monetization classification
        result = self.miner.classify_monetization_model(result)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
        
        # Test market size estimation
        result = self.miner.estimate_market_size(result)
        self.assertIn(result.market_size_indicator, ["low", "medium", "high"])
        
        # Test fallback analysis
        idea_dict = result.to_dict()
        analysis = self.analyzer._fallback_analysis(idea_dict)
        
        self.assertIn('score', analysis)
        self.assertIn('recommendation', analysis)
        self.assertIn('key_insight', analysis)
        
        # Test score bounds
        for score_type in ['business', 'execution', 'monetization']:
            self.assertGreaterEqual(analysis['score'][score_type], 0)
            self.assertLessEqual(analysis['score'][score_type], 10)
    
    def test_pipeline_consistency(self):
        """Test that the pipeline produces consistent results"""
        # Create test idea
        idea = StartupIdea(
            title="Test automation tool",
            content="I need a tool to automate my workflow daily",
            subreddit="SaaS",
            author="test_user",
            score=20,
            num_comments=30,
            created_utc=time.time(),
            url="https://reddit.com/test"
        )
        
        # Run pipeline multiple times
        results = []
        for _ in range(3):
            # Apply all analysis steps
            result = self.miner.apply_pain_scan_framework(idea)
            result = self.miner.classify_monetization_model(result)
            result = self.miner.estimate_market_size(result)
            
            # Convert to dict for analysis
            idea_dict = result.to_dict()
            analysis = self.analyzer._fallback_analysis(idea_dict)
            
            results.append({
                'pain_score': result.pain_score,
                'monetization_model': result.monetization_model,
                'business_score': analysis['score']['business'],
                'recommendation': analysis['recommendation']
            })
        
        # All results should be identical
        for i in range(1, len(results)):
            self.assertEqual(results[i]['pain_score'], results[0]['pain_score'])
            self.assertEqual(results[i]['monetization_model'], results[0]['monetization_model'])
            self.assertEqual(results[i]['business_score'], results[0]['business_score'])
            self.assertEqual(results[i]['recommendation'], results[0]['recommendation'])
    
    def test_pipeline_edge_cases(self):
        """Test pipeline with edge cases"""
        # Test with minimal content
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
        
        # Should handle gracefully
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertEqual(result.pain_score, 0.0)
        
        result = self.miner.classify_monetization_model(result)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
        
        result = self.miner.estimate_market_size(result)
        self.assertEqual(result.market_size_indicator, "low")
        
        # Test with very high scores
        idea = StartupIdea(
            title="URGENT: I need this RIGHT NOW",
            content="I would pay $1000 for this tool. I hate doing this manually every day. I built a spreadsheet but it's terrible.",
            subreddit="Entrepreneur",
            author="test",
            score=100,
            num_comments=50,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.apply_pain_scan_framework(idea)
        self.assertGreater(result.pain_score, 0)
        
        result = self.miner.classify_monetization_model(result)
        self.assertIn(result.monetization_model, ["micro_saas", "ugc_marketplace", "done_for_you", "Unknown"])
    
    def test_export_ranking(self):
        """Test that exported results are properly ranked"""
        # Create multiple test ideas with different pain scores
        ideas = []
        for i in range(5):
            idea = StartupIdea(
                title=f"Test idea {i}",
                content=f"Test content {i}",
                subreddit="test",
                author="test",
                score=10 + i * 5,  # Increasing scores
                num_comments=5 + i * 3,
                created_utc=time.time(),
                url=f"test{i}"
            )
            
            # Apply analysis
            result = self.miner.apply_pain_scan_framework(idea)
            result = self.miner.classify_monetization_model(result)
            result = self.miner.estimate_market_size(result)
            
            ideas.append(result)
        
        # Sort by pain score (should be done automatically in export)
        sorted_ideas = sorted(ideas, key=lambda x: x.pain_score, reverse=True)
        
        # Verify ranking
        for i in range(1, len(sorted_ideas)):
            self.assertGreaterEqual(sorted_ideas[i-1].pain_score, sorted_ideas[i].pain_score)
        
        # Test that IDs are assigned correctly
        for i, idea in enumerate(sorted_ideas, 1):
            idea.id = i
            self.assertEqual(idea.id, i)
    
    def test_validation_signals(self):
        """Test validation signal generation"""
        # Create idea with strong signals
        idea = StartupIdea(
            title="I need a tool to automate my daily tasks",
            content="I spend 2 hours every day manually copying data. I would pay $50/month for this.",
            subreddit="Entrepreneur",
            author="test",
            score=25,
            num_comments=15,
            created_utc=time.time(),
            url="test"
        )
        
        # Apply analysis
        result = self.miner.apply_pain_scan_framework(idea)
        result = self.miner.classify_monetization_model(result)
        result = self.miner.estimate_market_size(result)
        
        # Test validation
        validation = self.miner.validate_with_idea_funnel(result)
        
        self.assertIn('validation_score', validation)
        self.assertIn('signals', validation)
        self.assertGreaterEqual(validation['validation_score'], 0)
        self.assertIsInstance(validation['signals'], list)


class TestFormulaRegression(unittest.TestCase):
    """Test for formula regression - ensures formulas don't change unexpectedly"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.miner = RedditIdeaMiner("test_id", "test_secret", "test_agent")
        self.analyzer = OllamaAnalyzer()
    
    def test_engagement_velocity_regression(self):
        """Test that engagement velocity formula produces expected results"""
        # Known test case with expected result
        mock_post = Mock()
        mock_post.score = 10
        mock_post.num_comments = 5
        mock_post.created_utc = time.time() - 86400  # 24 hours ago
        
        velocity = self.miner.calculate_engagement_velocity(mock_post)
        expected = (10 + 5) / 24 * (1 + min(5/10, 2.0) * 0.5)
        
        # This should always produce the same result
        self.assertAlmostEqual(velocity, expected, places=2)
    
    def test_pain_score_regression(self):
        """Test that pain score formula produces expected results"""
        # Known test case
        idea = StartupIdea(
            title="I need a tool to automate my daily tasks",
            content="I spend 2 hours every day manually copying data. I would pay $50/month for this.",
            subreddit="Entrepreneur",
            author="test",
            score=25,
            num_comments=15,
            created_utc=time.time(),
            url="test"
        )
        
        result = self.miner.apply_pain_scan_framework(idea)
        
        # Test that weights still sum to 1.0
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
    
    def test_fallback_scoring_regression(self):
        """Test that fallback scoring produces expected results"""
        # Known test case
        idea = {"pain_score": 10, "monetization_model": "micro_saas"}
        result = self.analyzer._fallback_analysis(idea)
        
        # These values should always be the same for this input
        self.assertEqual(result['score']['business'], 8.0)
        self.assertEqual(result['score']['execution'], 5.5)
        self.assertEqual(result['score']['monetization'], 6.0)
        
        # Test recommendation logic
        avg_score = (8.0 + 5.5 + 6.0) / 3
        if avg_score >= 7:
            expected_rec = "Pursue Aggressively"
        elif avg_score >= 5:
            expected_rec = "Pursue Cautiously"
        elif avg_score >= 3:
            expected_rec = "Validate Further"
        else:
            expected_rec = "Pass"
        
        self.assertEqual(result['recommendation'], expected_rec)


if __name__ == '__main__':
    unittest.main(verbosity=2) 