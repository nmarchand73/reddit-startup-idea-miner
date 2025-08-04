#!/usr/bin/env python3
"""
RedditIdeaMiner class for mining startup ideas from Reddit
"""

import praw
import requests
import time
import json
import logging
import pandas as pd
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter, defaultdict
import numpy as np
from urllib.parse import quote
import re
import os

from startup_idea import StartupIdea
from constants import GOLDMINE_SUBREDDITS, PAIN_PATTERNS, MONETIZATION_MODELS, DEFAULT_KEYWORDS, OUTPUT_DIR

logger = logging.getLogger(__name__)

class RedditIdeaMiner:
    """Complete Reddit idea mining system implementing the 2026 methodology"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize with Reddit API credentials"""
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            # Test the connection
            self.reddit.user.me()
            logger.info("Successfully authenticated with Reddit API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Reddit API: {e}")
            raise ValueError(f"Invalid Reddit API credentials: {e}")
        
        self.ideas: List[StartupIdea] = []
        self.session = requests.Session()
        self.seen_hashes = set()  # For deduplication
        
    def calculate_engagement_velocity(self, post) -> float:
        """Track engagement velocity, not just subscriber count (Step 1)"""
        try:
            post_age_hours = (time.time() - post.created_utc) / 3600
            if post_age_hours <= 0:
                return 0.0
                
            # Base engagement velocity = (upvotes + comments) / age_in_hours
            base_velocity = (post.score + post.num_comments) / post_age_hours
            
            # Discussion bonus for high comment-to-upvote ratio (indicates discussion)
            discussion_bonus = 0.0
            if post.score > 0:
                comment_ratio = post.num_comments / post.score
                discussion_bonus = min(comment_ratio, 2.0) * 0.5
                
            # Apply discussion bonus
            velocity = base_velocity * (1 + discussion_bonus)
                
            return velocity
        except Exception as e:
            logger.error(f"Error calculating engagement velocity: {e}")
            return 0.0
    
    def apply_pain_scan_framework(self, idea: StartupIdea) -> StartupIdea:
        """Apply the 5-Point Pain Scan Framework (Step 2)"""
        text = f"{idea.title} {idea.content}".lower()
        
        # 1. Specificity Score
        specificity_score = 0.0
        for pattern in PAIN_PATTERNS['specificity_indicators']['patterns']:
            if re.search(pattern, text):
                specificity_score += PAIN_PATTERNS['specificity_indicators']['weight']
        
        # 2. Frequency Score
        frequency_score = 0.0
        for keyword in PAIN_PATTERNS['frequency_indicators']['keywords']:
            if keyword in text:
                frequency_score += PAIN_PATTERNS['frequency_indicators']['weight']
        
        # 3. DIY Evidence Score
        diy_score = 0.0
        for keyword in PAIN_PATTERNS['diy_evidence']['keywords']:
            if keyword in text:
                diy_score += PAIN_PATTERNS['diy_evidence']['weight']
        
        # 4. Money Signals Score
        money_score = 0.0
        for pattern in PAIN_PATTERNS['money_signals']['patterns']:
            if re.search(pattern, text):
                money_score += PAIN_PATTERNS['money_signals']['weight']
        
        for keyword in PAIN_PATTERNS['money_signals']['keywords']:
            if keyword in text:
                money_score += PAIN_PATTERNS['money_signals']['weight']
        
        # 5. Group Endorsement Score (based on upvotes and comments)
        endorsement_score = 0.0
        if idea.score > 10:
            endorsement_score += 2.0
        elif idea.score > 5:
            endorsement_score += 1.0
        
        if idea.num_comments > 20:
            endorsement_score += 2.0
        elif idea.num_comments > 10:
            endorsement_score += 1.0
        
        # Calculate total pain score
        idea.specificity_score = specificity_score
        idea.frequency_score = frequency_score
        idea.diy_score = diy_score
        idea.money_score = money_score
        idea.endorsement_score = endorsement_score
        
        # Weighted pain score calculation
        idea.pain_score = (
            specificity_score * 0.2 +
            frequency_score * 0.2 +
            diy_score * 0.25 +
            money_score * 0.25 +
            endorsement_score * 0.1
        )
        
        return idea
    
    def classify_monetization_model(self, idea: StartupIdea) -> StartupIdea:
        """Classify monetization model based on content analysis"""
        text = f"{idea.title} {idea.content}".lower()
        
        best_model = "Unknown"
        best_score = 0.0
        
        for model_name, model_data in MONETIZATION_MODELS.items():
            score = 0.0
            
            # Check for signals
            for signal in model_data['signals']:
                if signal in text:
                    score += model_data['weight']
            
            # Check for Reddit-specific signals
            if model_data.get('reddit_signal', '').lower() in text:
                score += model_data['weight'] * 1.5
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        idea.monetization_model = best_model
        return idea
    
    def estimate_market_size(self, idea: StartupIdea) -> StartupIdea:
        """Estimate market size based on pain score and engagement"""
        if idea.pain_score >= 20:
            idea.market_size_indicator = "high"
        elif idea.pain_score >= 15:
            idea.market_size_indicator = "medium"
        else:
            idea.market_size_indicator = "low"
        
        # Competition level based on subreddit
        if idea.subreddit in ['Entrepreneur', 'SaaS']:
            idea.competition_level = "high"
        elif idea.subreddit in ['startups', 'smallbusiness']:
            idea.competition_level = "medium"
        else:
            idea.competition_level = "low"
        
        return idea
    
    def is_valid_post(self, post) -> bool:
        """Check if a post is valid for idea mining"""
        try:
            # Skip deleted/removed posts
            if post.author is None or post.selftext == '[deleted]' or post.selftext == '[removed]':
                return False
            
            # Skip posts with no content
            if not post.selftext or len(post.selftext.strip()) < 50:
                return False
            
            # Skip posts that are too old (more than 60 days)
            post_age_days = (time.time() - post.created_utc) / 86400
            if post_age_days > 60:
                return False
            
            # Skip posts with very low engagement
            if post.score < 2 and post.num_comments < 3:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating post: {e}")
            return False
    
    def is_duplicate_idea(self, idea: StartupIdea) -> bool:
        """Check if idea is a duplicate based on content hash"""
        return idea.content_hash in self.seen_hashes
    
    def search_goldmine_subreddits(self, keywords: List[str], days_back: int = 60) -> List[StartupIdea]:
        """Search the 7 goldmine subreddits for startup ideas"""
        ideas = []
        
        for subreddit_name, config in GOLDMINE_SUBREDDITS.items():
            try:
                logger.info(f"Searching r/{subreddit_name}...")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search with keywords
                for keyword in keywords:
                    search_results = subreddit.search(keyword, sort='hot', time_filter='month', limit=50)
                    
                    for post in search_results:
                        if not self.is_valid_post(post):
                            continue
                        
                        # Create StartupIdea object
                        idea = StartupIdea(
                            title=post.title,
                            content=post.selftext,
                            subreddit=subreddit_name,
                            author=str(post.author),
                            score=post.score,
                            num_comments=post.num_comments,
                            created_utc=post.created_utc,
                            url=f"https://reddit.com{post.permalink}"
                        )
                        
                        # Skip duplicates
                        if self.is_duplicate_idea(idea):
                            continue
                        
                        # Apply analysis framework
                        idea.engagement_velocity = self.calculate_engagement_velocity(post)
                        idea = self.apply_pain_scan_framework(idea)
                        idea = self.classify_monetization_model(idea)
                        idea = self.estimate_market_size(idea)
                        
                        # Add to collection
                        ideas.append(idea)
                        self.seen_hashes.add(idea.content_hash)
                        
                        # Rate limiting
                        time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error searching r/{subreddit_name}: {e}")
                continue
        
        return ideas
    
    def apply_dark_funnel_tactics(self, keywords: List[str]) -> List[StartupIdea]:
        """Apply dark funnel tactics (simulated)"""
        # This is a simulation of dark funnel tactics
        # In a real implementation, you might use different techniques
        logger.info("Applying dark funnel tactics (simulated)...")
        
        ideas = []
        for keyword in keywords:
            # Simulate finding hidden pain points
            simulated_idea = StartupIdea(
                title=f"Hidden pain point: {keyword}",
                content=f"Simulated dark funnel discovery for {keyword}",
                subreddit="dark_funnel",
                author="dark_funnel_bot",
                score=15,
                num_comments=8,
                created_utc=time.time() - 86400,  # 1 day ago
                url="https://reddit.com/r/dark_funnel/simulated"
            )
            
            simulated_idea.engagement_velocity = 0.5
            simulated_idea = self.apply_pain_scan_framework(simulated_idea)
            simulated_idea = self.classify_monetization_model(simulated_idea)
            simulated_idea = self.estimate_market_size(simulated_idea)
            
            ideas.append(simulated_idea)
        
        return ideas
    
    def validate_with_idea_funnel(self, idea: StartupIdea) -> Dict:
        """Validate idea using the idea funnel system"""
        validation_signals = []
        
        # Check for validation signals
        if idea.pain_score >= 20:
            validation_signals.append("Exceptional pain score")
        
        if idea.engagement_velocity > 0.5:
            validation_signals.append("High engagement velocity")
        
        if idea.monetization_model != "Unknown":
            validation_signals.append(f"Clear monetization model: {idea.monetization_model}")
        
        if idea.is_problem_statement:
            validation_signals.append("Clear problem statement")
        
        if idea.money_score > 3:
            validation_signals.append("Strong money signals")
        
        idea.validation_signals = validation_signals
        
        return {
            "idea": idea,
            "validation_score": len(validation_signals),
            "signals": validation_signals
        }
    
    def mine_startup_ideas(self, keywords: List[str] = None, min_pain_score: float = 3.0, 
                          include_dark_funnel: bool = True) -> List[StartupIdea]:
        """Main method to mine startup ideas"""
        if keywords is None:
            keywords = DEFAULT_KEYWORDS
        
        logger.info(f"Starting idea mining with {len(keywords)} keywords...")
        
        # Step 1: Search goldmine subreddits
        ideas = self.search_goldmine_subreddits(keywords)
        logger.info(f"Found {len(ideas)} ideas from goldmine subreddits")
        
        # Step 2: Apply dark funnel tactics (optional)
        if include_dark_funnel:
            dark_ideas = self.apply_dark_funnel_tactics(keywords)
            ideas.extend(dark_ideas)
            logger.info(f"Added {len(dark_ideas)} ideas from dark funnel")
        
        # Step 3: Apply validation funnel
        validated_ideas = []
        for idea in ideas:
            validation_result = self.validate_with_idea_funnel(idea)
            if validation_result['validation_score'] >= 2:  # At least 2 validation signals
                validated_ideas.append(idea)
        
        # Step 4: Filter by pain score
        filtered_ideas = [idea for idea in validated_ideas if idea.pain_score >= min_pain_score]
        
        # Step 5: Sort by pain score and engagement
        filtered_ideas.sort(key=lambda x: (x.pain_score, x.engagement_velocity), reverse=True)
        
        self.ideas = filtered_ideas
        logger.info(f"Mining complete. Found {len(self.ideas)} validated ideas with pain score >= {min_pain_score}")
        
        return self.ideas
    
    def generate_validation_report(self, top_n: int = 10) -> Dict:
        """Generate a comprehensive validation report"""
        if not self.ideas:
            return {"error": "No ideas to analyze"}
        
        top_ideas = self.ideas[:top_n]
        
        # Calculate statistics
        pain_scores = [idea.pain_score for idea in self.ideas]
        engagement_velocities = [idea.engagement_velocity for idea in self.ideas]
        
        # Monetization breakdown
        monetization_counts = Counter([idea.monetization_model for idea in self.ideas])
        
        # Subreddit breakdown
        subreddit_counts = Counter([idea.subreddit for idea in self.ideas])
        
        # High potential ideas (pain score >= 20)
        high_potential = [idea for idea in self.ideas if idea.pain_score >= 20]
        
        report = {
            "summary": {
                "total_ideas": len(self.ideas),
                "high_potential_count": len(high_potential),
                "average_pain_score": np.mean(pain_scores) if pain_scores else 0,
                "average_engagement_velocity": np.mean(engagement_velocities) if engagement_velocities else 0,
                "top_pain_score": max(pain_scores) if pain_scores else 0
            },
            "monetization_breakdown": dict(monetization_counts),
            "subreddit_breakdown": dict(subreddit_counts),
            "top_ideas": [idea.to_dict() for idea in top_ideas],
            "high_potential_ideas": [idea.to_dict() for idea in high_potential],
            "validation_insights": {
                "problem_statements": len([idea for idea in self.ideas if idea.is_problem_statement]),
                "success_stories": len([idea for idea in self.ideas if idea.is_success_story]),
                "clear_monetization": len([idea for idea in self.ideas if idea.monetization_model != "Unknown"])
            }
        }
        
        return report
    
    def export_results(self, format: str = 'csv', filename: str = None) -> str:
        """Export the mined ideas to a CSV or JSON file."""
        if not self.ideas:
            raise ValueError("No ideas to export. Run mine_startup_ideas() first.")
        
        # Sort ideas by pain score (highest first) and add unique IDs
        sorted_ideas = sorted(self.ideas, key=lambda x: x.pain_score, reverse=True)
        
        # Add unique ID to each idea
        for i, idea in enumerate(sorted_ideas, 1):
            idea.id = i
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"startup_ideas_{timestamp}"
        
        # Prepend output directory to filename
        filename = os.path.join(OUTPUT_DIR, filename)
        
        if format.lower() == 'csv':
            filename += '.csv'
            try:
                # Convert ideas to list of dictionaries with ID
                ideas_data = []
                for idea in sorted_ideas:
                    idea_dict = idea.to_dict()
                    idea_dict['id'] = idea.id  # Add the unique ID
                    ideas_data.append(idea_dict)
                
                df = pd.DataFrame(ideas_data)
                
                # Reorder columns to put ID first, then pain_score, then other fields
                priority_columns = ['id', 'pain_score', 'title', 'subreddit', 'monetization_model']
                other_columns = [col for col in df.columns if col not in priority_columns]
                columns = priority_columns + other_columns
                df = df[columns]
                
                df.to_csv(filename, index=False, encoding='utf-8')
                print(f"Exported {len(sorted_ideas)} ideas to {filename} (ranked by pain_score)")
                
            except Exception as e:
                print(f"Error exporting to CSV: {e}")
                raise
                
        elif format.lower() == 'json':
            filename += '.json'
            try:
                # Convert ideas to list of dictionaries with ID
                ideas_data = []
                for idea in sorted_ideas:
                    idea_dict = idea.to_dict()
                    idea_dict['id'] = idea.id  # Add the unique ID
                    ideas_data.append(idea_dict)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(ideas_data, f, indent=2, ensure_ascii=False)
                print(f"Exported {len(sorted_ideas)} ideas to {filename} (ranked by pain_score)")
                
            except Exception as e:
                print(f"Error exporting to JSON: {e}")
                raise
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filename 