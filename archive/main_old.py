#!/usr/bin/env python3
"""
The Ultimate Reddit Idea Sourcing Automation - 2026 Edition
Complete implementation of Marshall Hargrave's methodology for mining $100k+ startup concepts.

Features:
- All 5 steps of the methodology
- Dark Funnel tactics (deleted threads, comment archaeology)
- Engagement velocity tracking
- 5-Point Pain Scan Framework
- Monetization mapping
- Validation system
- Ollama AI analysis
"""

import praw
import pandas as pd
import requests
import re
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from urllib.parse import quote
import csv
from pathlib import Path
import sys
import os
import hashlib
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StartupIdea:
    """Complete data structure for startup ideas with validation metrics"""
    title: str
    content: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    
    # Validation metrics
    pain_score: float = 0.0
    specificity_score: float = 0.0
    frequency_score: float = 0.0
    diy_score: float = 0.0
    money_score: float = 0.0
    endorsement_score: float = 0.0
    
    # Business potential
    monetization_model: str = ""
    market_size_indicator: str = ""
    competition_level: str = ""
    validation_signals: List[str] = None
    
    # Engagement metrics
    engagement_velocity: float = 0.0
    comment_quality_score: float = 0.0
    
    # Content quality
    is_problem_statement: bool = False
    is_success_story: bool = False
    content_hash: str = ""
    
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
        
        # Success story indicators
        success_indicators = [
            'i made', 'i earned', 'i built', 'i launched', 'i sold', 'i scaled',
            'success story', 'how i', 'my journey', 'from 0 to', 'reached',
            'achieved', 'hit', 'made $', 'earned $', 'revenue', 'mrr', 'arr'
        ]
        
        # Problem statement indicators
        problem_indicators = [
            'i need', 'i want', 'i wish', 'someone should', 'there should be',
            'why doesn\'t', 'frustrated', 'annoyed', 'hate', 'sucks', 'terrible',
            'problem with', 'issue with', 'struggling', 'difficult', 'hard',
            'looking for', 'anyone know', 'help me', 'advice needed'
        ]
        
        success_count = sum(1 for indicator in success_indicators if indicator in text)
        problem_count = sum(1 for indicator in problem_indicators if indicator in text)
        
        self.is_success_story = success_count > problem_count
        self.is_problem_statement = problem_count > success_count


class OllamaAnalyzer:
    """AI-powered analysis of startup ideas using Ollama"""
    
    def __init__(self, model: str = "phi4-mini", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def analyze_idea(self, idea: Dict) -> Dict:
        """Analyze a single startup idea using Ollama"""
        if not self._check_ollama_available():
            return {"error": "Ollama not available. Please start Ollama service."}
        
        # Check if the specified model is available
        available_models = self._get_available_models()
        if not available_models:
            return {"error": "No Ollama models found. Please pull a model: ollama pull phi4-mini"}
        
        # Use the first available model if the specified one isn't available
        if self.model not in available_models:
            print(f"Warning: Model '{self.model}' not found. Using '{available_models[0]}' instead.")
            self.model = available_models[0]
        
        prompt = self._create_analysis_prompt(idea)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30  # Reduced timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_analysis_result(result['response'], idea)
            else:
                return {"error": f"Ollama API error: {response.status_code} - {response.text}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Analysis timed out. Try with a smaller model or check Ollama performance."}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _fallback_analysis(self, idea: Dict) -> Dict:
        """Provide a comprehensive fallback analysis when Ollama is unavailable"""
        pain_score = idea.get('pain_score', 0)
        engagement_velocity = idea.get('engagement_velocity', 0)
        monetization_model = idea.get('monetization_model', 'Unknown')
        title = idea.get('title', 'N/A')
        content = idea.get('content_preview', 'N/A')
        
        # Business potential scoring based on pain score
        if pain_score >= 20:
            business_score = 8
            business_reasoning = f"Exceptional pain score ({pain_score}) indicates strong market validation and urgent need"
            urgency_level = "high"
            market_size = "large"
        elif pain_score >= 15:
            business_score = 7
            business_reasoning = f"High pain score ({pain_score}) indicates validated market need"
            urgency_level = "medium"
            market_size = "medium"
        elif pain_score >= 10:
            business_score = 6
            business_reasoning = f"Moderate pain score ({pain_score}) indicates some market need"
            urgency_level = "medium"
            market_size = "medium"
        else:
            business_score = 4
            business_reasoning = f"Low pain score ({pain_score}) indicates limited market validation"
            urgency_level = "low"
            market_size = "small"
        
        # Execution viability based on content analysis
        execution_score = 6
        challenges = ["Standard implementation challenges"]
        time_to_market = "months"
        resource_requirements = "medium"
        
        # Monetization scoring
        if monetization_model == "micro_saas":
            monetization_score = 8
            strategy = "SaaS subscription model with recurring revenue"
            scalability = "high"
            pricing_power = "medium"
        elif monetization_model == "ugc_marketplace":
            monetization_score = 7
            strategy = "Marketplace model with transaction fees"
            scalability = "high"
            pricing_power = "low"
        elif monetization_model == "done_for_you":
            monetization_score = 6
            strategy = "Service-based model with project pricing"
            scalability = "medium"
            pricing_power = "medium"
        else:
            monetization_score = 5
            strategy = "Undefined monetization model"
            scalability = "unknown"
            pricing_power = "unknown"
        
        # Risk assessment
        primary_risks = ["Market competition", "Execution complexity"]
        mitigation_strategies = ["Focus on unique value proposition", "Start with MVP approach"]
        
        # Recommendation logic
        total_score = (business_score + execution_score + monetization_score) / 3
        if total_score >= 7:
            recommendation = "Pursue Aggressively"
        elif total_score >= 5:
            recommendation = "Pursue Cautiously"
        elif total_score >= 3:
            recommendation = "Validate Further"
        else:
            recommendation = "Pass"
        
        # Next steps based on recommendation
        if recommendation == "Pursue Aggressively":
            next_steps = [
                "Conduct detailed market research",
                "Build MVP within 2-3 months",
                "Start customer interviews immediately"
            ]
        elif recommendation == "Pursue Cautiously":
            next_steps = [
                "Validate with 10-20 potential customers",
                "Research competitive landscape",
                "Define MVP scope and timeline"
            ]
        elif recommendation == "Validate Further":
            next_steps = [
                "Gather more market data",
                "Interview potential customers",
                "Refine problem statement"
            ]
        else:
            next_steps = [
                "Consider alternative opportunities",
                "Reassess market approach",
                "Look for stronger pain signals"
            ]
        
        return {
            "business_potential": {
                "score": business_score,
                "reasoning": business_reasoning,
                "market_size": market_size,
                "urgency_level": urgency_level
            },
            "execution_viability": {
                "score": execution_score,
                "challenges": challenges,
                "time_to_market": time_to_market,
                "resource_requirements": resource_requirements
            },
            "monetization": {
                "score": monetization_score,
                "strategy": strategy,
                "scalability": scalability,
                "pricing_power": pricing_power
            },
            "risk_assessment": {
                "primary_risks": primary_risks,
                "mitigation_strategies": mitigation_strategies
            },
            "recommendation": recommendation,
            "next_steps": next_steps,
            "key_insight": f"Pain score {pain_score} with {monetization_model} model suggests {'strong' if pain_score >= 15 else 'moderate'} market opportunity",
            "fallback": True
        }
    
    def _create_analysis_prompt(self, idea: Dict) -> str:
        """Create a detailed analysis prompt for the startup idea"""
        pain_score = idea.get('pain_score', 0)
        engagement_velocity = idea.get('engagement_velocity', 0)
        monetization_model = idea.get('monetization_model', 'Unknown')
        subreddit = idea.get('subreddit', 'Unknown')
        
        # Context about pain scores
        pain_context = ""
        if pain_score >= 20:
            pain_context = "EXCEPTIONAL pain score - indicates strong market validation and urgent need"
        elif pain_score >= 15:
            pain_context = "HIGH pain score - indicates validated market need"
        elif pain_score >= 10:
            pain_context = "MODERATE pain score - indicates some market need"
        else:
            pain_context = "LOW pain score - limited market validation"
        
        # Engagement context
        engagement_context = ""
        if engagement_velocity > 0.5:
            engagement_context = "HIGH engagement - strong community interest"
        elif engagement_velocity > 0.2:
            engagement_context = "MODERATE engagement - decent community interest"
        else:
            engagement_context = "LOW engagement - limited community interest"
        
        return f"""
You are a startup idea analyst. Analyze this Reddit post as a potential business opportunity.

CONTEXT:
- PAIN SCORE: {pain_score} ({pain_context})
- ENGAGEMENT: {engagement_velocity:.3f} ({engagement_context})
- MONETIZATION MODEL: {monetization_model}
- SUBREDDIT: r/{subreddit}
- TITLE: {idea.get('title', 'N/A')}
- CONTENT: {idea.get('content_preview', 'N/A')}

ANALYSIS FRAMEWORK:
1. Business Potential (1-10): Market size, demand urgency, competitive landscape
2. Execution Viability (1-10): Technical feasibility, resource requirements, time to market
3. Monetization Strength (1-10): Revenue model clarity, pricing power, scalability
4. Risk Assessment: Key challenges and mitigation strategies
5. Next Steps: Specific actionable recommendations

Provide analysis in this JSON format:
{{
    "business_potential": {{
        "score": 1-10,
        "reasoning": "Specific market opportunity and demand analysis",
        "market_size": "small/medium/large",
        "urgency_level": "low/medium/high"
    }},
    "execution_viability": {{
        "score": 1-10,
        "challenges": ["specific challenge 1", "specific challenge 2"],
        "time_to_market": "weeks/months/years",
        "resource_requirements": "low/medium/high"
    }},
    "monetization": {{
        "score": 1-10,
        "strategy": "Specific revenue approach with pricing model",
        "scalability": "low/medium/high",
        "pricing_power": "low/medium/high"
    }},
    "risk_assessment": {{
        "primary_risks": ["risk 1", "risk 2"],
        "mitigation_strategies": ["strategy 1", "strategy 2"]
    }},
    "recommendation": "Pursue Aggressively/Pursue Cautiously/Validate Further/Pass",
    "next_steps": ["specific action 1", "specific action 2", "specific action 3"],
    "key_insight": "Most important business insight or opportunity"
}}

Guidelines:
- Pain scores 20+ indicate exceptional market validation
- High engagement velocity suggests strong community interest
- Be specific about market opportunities and execution challenges
- Provide actionable next steps, not generic advice
- Consider the monetization model in your analysis
"""
    
    def _parse_analysis_result(self, response: str, original_idea: Dict) -> Dict:
        """Parse the Ollama response and extract structured analysis"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                analysis['original_idea'] = original_idea
                return analysis
            else:
                return {
                    "error": "Could not parse JSON response",
                    "raw_response": response,
                    "original_idea": original_idea
                }
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON response",
                "raw_response": response,
                "original_idea": original_idea
            }
    
    def analyze_ideas_batch(self, ideas: List[Dict], max_ideas: int = 10) -> Dict:
        """Analyze multiple ideas and provide batch insights"""
        # Limit analysis to top ideas
        top_ideas = sorted(ideas, key=lambda x: x.get('pain_score', 0), reverse=True)[:max_ideas]
        
        # Check if Ollama is available
        if not self._check_ollama_available():
            print("Ollama not available. Using fallback analysis...")
            analyses = []
            for i, idea in enumerate(top_ideas, 1):
                print(f"Analyzing idea {i}/{len(top_ideas)}: {idea.get('title', 'N/A')[:50]}...")
                analysis = self._fallback_analysis(idea)
                analyses.append(analysis)
                time.sleep(0.1)  # Small delay
            
            return {
                "analyses": analyses,
                "summary": self._create_batch_summary(analyses),
                "recommendations": self._create_batch_recommendations(analyses),
                "fallback_used": True
            }
        
        analyses = []
        for i, idea in enumerate(top_ideas, 1):
            print(f"Analyzing idea {i}/{len(top_ideas)}: {idea.get('title', 'N/A')[:50]}...")
            analysis = self.analyze_idea(idea)
            
            # If Ollama analysis fails, use fallback
            if 'error' in analysis:
                print(f"  Ollama analysis failed, using fallback...")
                analysis = self._fallback_analysis(idea)
            
            analyses.append(analysis)
            time.sleep(1)  # Rate limiting
        
        return {
            "analyses": analyses,
            "summary": self._create_batch_summary(analyses),
            "recommendations": self._create_batch_recommendations(analyses)
        }
    
    def _create_batch_summary(self, analyses: List[Dict]) -> Dict:
        """Create a comprehensive summary of batch analysis results"""
        if not analyses:
            return {"error": "No analyses to summarize"}
        
        # Extract scores and categorize recommendations
        business_scores = []
        execution_scores = []
        monetization_scores = []
        pursue_aggressively = []
        pursue_cautiously = []
        validate_further = []
        pass_ideas = []
        
        for analysis in analyses:
            if 'business_potential' in analysis and 'score' in analysis['business_potential']:
                business_scores.append(analysis['business_potential']['score'])
            if 'execution_viability' in analysis and 'score' in analysis['execution_viability']:
                execution_scores.append(analysis['execution_viability']['score'])
            if 'monetization' in analysis and 'score' in analysis['monetization']:
                monetization_scores.append(analysis['monetization']['score'])
            
            # Categorize by recommendation
            recommendation = analysis.get('recommendation', 'Unknown')
            if 'Aggressively' in recommendation:
                pursue_aggressively.append(analysis)
            elif 'Cautiously' in recommendation:
                pursue_cautiously.append(analysis)
            elif 'Validate' in recommendation:
                validate_further.append(analysis)
            elif 'Pass' in recommendation:
                pass_ideas.append(analysis)
        
        # Calculate averages
        avg_business = sum(business_scores) / len(business_scores) if business_scores else 0
        avg_execution = sum(execution_scores) / len(execution_scores) if execution_scores else 0
        avg_monetization = sum(monetization_scores) / len(monetization_scores) if monetization_scores else 0
        
        # Find top scoring ideas
        top_ideas = sorted(analyses, 
                          key=lambda x: (x.get('business_potential', {}).get('score', 0) + 
                                        x.get('execution_viability', {}).get('score', 0) + 
                                        x.get('monetization', {}).get('score', 0)) / 3, 
                          reverse=True)[:3]
        
        # Market opportunity analysis
        market_opportunities = []
        for analysis in analyses:
            if 'business_potential' in analysis:
                bp = analysis['business_potential']
                if bp.get('urgency_level') == 'high' and bp.get('market_size') == 'large':
                    market_opportunities.append({
                        'title': analysis.get('original_idea', {}).get('title', 'Unknown'),
                        'pain_score': analysis.get('original_idea', {}).get('pain_score', 0),
                        'reasoning': bp.get('reasoning', '')
                    })
        
        return {
            "total_ideas_analyzed": len(analyses),
            "average_scores": {
                "business_potential": round(avg_business, 2),
                "execution_viability": round(avg_execution, 2),
                "monetization": round(avg_monetization, 2),
                "overall": round((avg_business + avg_execution + avg_monetization) / 3, 2)
            },
            "recommendation_distribution": {
                "pursue_aggressively": len(pursue_aggressively),
                "pursue_cautiously": len(pursue_cautiously),
                "validate_further": len(validate_further),
                "pass": len(pass_ideas)
            },
            "top_scoring_ideas": top_ideas,
            "high_opportunity_ideas": market_opportunities,
            "market_insights": {
                "high_pain_ideas": len([a for a in analyses if a.get('original_idea', {}).get('pain_score', 0) >= 20]),
                "high_engagement_ideas": len([a for a in analyses if a.get('original_idea', {}).get('engagement_velocity', 0) > 0.5]),
                "saas_opportunities": len([a for a in analyses if a.get('original_idea', {}).get('monetization_model') == 'micro_saas'])
            }
        }
    
    def _create_batch_recommendations(self, analyses: List[Dict]) -> Dict:
        """Create actionable batch recommendations based on analysis results"""
        if not analyses:
            return {"error": "No analyses for recommendations"}
        
        # Collect common themes and insights
        common_challenges = []
        common_opportunities = []
        monetization_models = {}
        subreddit_insights = {}
        
        for analysis in analyses:
            original_idea = analysis.get('original_idea', {})
            
            # Track monetization models
            model = original_idea.get('monetization_model', 'Unknown')
            monetization_models[model] = monetization_models.get(model, 0) + 1
            
            # Track subreddit insights
            subreddit = original_idea.get('subreddit', 'Unknown')
            if subreddit not in subreddit_insights:
                subreddit_insights[subreddit] = {
                    'count': 0,
                    'avg_pain_score': 0,
                    'high_potential_count': 0
                }
            subreddit_insights[subreddit]['count'] += 1
            subreddit_insights[subreddit]['avg_pain_score'] += original_idea.get('pain_score', 0)
            
            # Track high potential ideas by subreddit
            if analysis.get('recommendation', '').startswith('Pursue'):
                subreddit_insights[subreddit]['high_potential_count'] += 1
            
            # Collect challenges and opportunities
            if 'execution_viability' in analysis and 'challenges' in analysis['execution_viability']:
                common_challenges.extend(analysis['execution_viability']['challenges'])
            
            if 'business_potential' in analysis and analysis['business_potential'].get('urgency_level') == 'high':
                common_opportunities.append({
                    'title': original_idea.get('title', 'Unknown'),
                    'pain_score': original_idea.get('pain_score', 0),
                    'reasoning': analysis['business_potential'].get('reasoning', '')
                })
        
        # Calculate subreddit averages
        for subreddit in subreddit_insights:
            if subreddit_insights[subreddit]['count'] > 0:
                subreddit_insights[subreddit]['avg_pain_score'] /= subreddit_insights[subreddit]['count']
        
        # Generate strategic recommendations
        strategic_recommendations = []
        
        # High pain score opportunities
        high_pain_ideas = [a for a in analyses if a.get('original_idea', {}).get('pain_score', 0) >= 20]
        if high_pain_ideas:
            strategic_recommendations.append({
                "type": "high_pain_opportunity",
                "description": f"Found {len(high_pain_ideas)} ideas with exceptional pain scores (20+)",
                "action": "Prioritize these for immediate validation and MVP development"
            })
        
        # Monetization model insights
        if monetization_models:
            top_model = max(monetization_models.items(), key=lambda x: x[1])
            strategic_recommendations.append({
                "type": "monetization_trend",
                "description": f"Most common monetization model: {top_model[0]} ({top_model[1]} ideas)",
                "action": f"Focus on {top_model[0]} business models for faster market entry"
            })
        
        # Subreddit insights
        best_subreddit = max(subreddit_insights.items(), key=lambda x: x[1]['high_potential_count'])
        if best_subreddit[1]['high_potential_count'] > 0:
            strategic_recommendations.append({
                "type": "subreddit_opportunity",
                "description": f"r/{best_subreddit[0]} has highest potential ideas ({best_subreddit[1]['high_potential_count']})",
                "action": f"Focus more scraping efforts on r/{best_subreddit[0]} for better opportunities"
            })
        
        return {
            "strategic_recommendations": strategic_recommendations,
            "monetization_breakdown": monetization_models,
            "subreddit_insights": subreddit_insights,
            "common_challenges": list(set(common_challenges))[:5],  # Top 5 unique challenges
            "high_opportunity_ideas": common_opportunities[:3],  # Top 3 opportunities
            "next_actions": [
                "Validate top 3 ideas with customer interviews",
                "Build MVP for highest pain score opportunity",
                "Focus on most promising monetization model",
                "Increase scraping from highest-potential subreddits"
            ]
        }


class RedditIdeaMiner:
    """Complete Reddit idea mining system implementing the 2026 methodology"""
    
    # Step 1: The 7 Goldmine Subreddits (from methodology)
    GOLDMINE_SUBREDDITS = {
        'SomebodyMakeThis': {'weight': 3.0, 'focus': 'direct_requests'},
        'Entrepreneur': {'weight': 2.5, 'focus': 'business_pain'},
        'Technology': {'weight': 2.0, 'focus': 'trend_spotting'},
        'EntrepreneurRideAlong': {'weight': 2.8, 'focus': 'execution_proof'},
        'startups': {'weight': 2.3, 'focus': 'validation'},
        'SaaS': {'weight': 2.7, 'focus': 'software_needs'},
        'smallbusiness': {'weight': 2.4, 'focus': 'operational_pain'}
    }
    
    # Enhanced pain detection patterns based on methodology
    PAIN_PATTERNS = {
        'high_intensity': {
            'keywords': ['hate', 'sucks', 'terrible', 'awful', 'frustrated', 'annoying', 'waste of time', 'impossible', 'nightmare'],
            'weight': 3.0
        },
        'medium_intensity': {
            'keywords': ['difficult', 'hard', 'challenging', 'problem', 'issue', 'struggle', 'annoyed'],
            'weight': 2.0
        },
        'solution_seeking': {
            'keywords': ['need', 'want', 'wish', 'looking for', 'anyone know', 'help me find', 'someone should make'],
            'weight': 2.5
        },
        'specificity_indicators': {
            'patterns': [r'I need \w+ to \w+', r'specifically \w+', r'exactly \w+', r'in \w+ context'],
            'weight': 2.0
        },
        'frequency_indicators': {
            'keywords': ['every day', 'daily', 'always', 'constantly', 'every time', 'repeatedly', 'every damn', 'each time'],
            'weight': 2.5
        },
        'diy_evidence': {
            'keywords': ['built', 'created', 'made', 'spreadsheet', 'google sheet', 'manual', 'hack', 'workaround', 'custom solution'],
            'weight': 3.0
        },
        'money_signals': {
            'patterns': [r'\$\d+', r'pay.*\$', r'worth.*\$', r'invest.*\$', r'RIGHT NOW'],
            'keywords': ['pay', 'buy', 'purchase', 'invest', 'worth paying', 'would pay', 'happy to pay'],
            'weight': 4.0
        }
    }
    
    # Monetization model patterns from methodology
    MONETIZATION_MODELS = {
        'micro_saas': {
            'signals': ['automate', 'tool', 'dashboard', 'analytics', 'integration', 'api', 'tedious task'],
            'reddit_signal': 'I\'d pay $10/mo to automate',
            'weight': 3.0
        },
        'ugc_marketplace': {
            'signals': ['connect', 'find', 'marketplace', 'platform', 'freelance', 'creators', 'no platform for'],
            'reddit_signal': 'No platform for X creators',
            'weight': 2.5
        },
        'done_for_you': {
            'signals': ['template', 'done for you', 'service', 'agency', 'consulting', 'pre-made'],
            'reddit_signal': 'I\'d buy a pre-made template',
            'weight': 2.8
        }
    }
    
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
                
            # Engagement velocity = (upvotes + comments) / age_in_hours
            velocity = (post.score + post.num_comments) / post_age_hours
            
            # Bonus for high comment-to-upvote ratio (indicates discussion)
            if post.score > 0:
                discussion_bonus = min(post.num_comments / post.score, 2.0)
                velocity *= (1 + discussion_bonus * 0.5)
                
            return max(velocity, 0.0)  # Ensure non-negative
        except (AttributeError, ZeroDivisionError, TypeError) as e:
            logger.warning(f"Error calculating engagement velocity: {e}")
            return 0.0
    
    def apply_pain_scan_framework(self, idea: StartupIdea) -> StartupIdea:
        """Step 2: Apply the 5-Point Pain Scan Framework"""
        # Ensure content is not None and convert to string
        content = str(idea.content) if idea.content is not None else ""
        title = str(idea.title) if idea.title is not None else ""
        text = f"{title} {content}".lower()
        
        # 1. Specificity Test - Enhanced patterns
        specificity_patterns = [
            r'i need \w+ to \w+ \w+',
            r'specifically \w+',
            r'exactly \w+',
            r'in .+ context',
            r'for .+ use case',
            r'when .+ happens',
            r'whenever .+ occurs',
            r'if .+ then .+',
            r'because .+ i need',
            r'my .+ problem is'
        ]
        specificity_score = sum(len(re.findall(pattern, text)) for pattern in specificity_patterns)
        idea.specificity_score = min(specificity_score * 0.8, 3.0)  # Increased multiplier
        
        if specificity_score > 0:
            idea.validation_signals.append('specificity')
        
        # 2. Frequency Stamp - Enhanced indicators
        frequency_indicators = [
            'every day', 'daily', 'always', 'constantly', 'every damn', 'each time',
            'repeatedly', 'over and over', 'time and time again', 'consistently',
            'regularly', 'frequently', 'often', 'multiple times'
        ]
        frequency_score = sum(text.count(indicator) for indicator in frequency_indicators)
        idea.frequency_score = min(frequency_score * 1.2, 3.0)  # Increased multiplier
        
        if frequency_score > 0:
            idea.validation_signals.append('frequency')
        
        # 3. DIY Evidence - Enhanced keywords
        diy_keywords = [
            'built', 'created', 'made', 'spreadsheet', 'google sheet', 'manual', 
            'hack', 'workaround', 'custom solution', 'diy', 'homemade',
            'put together', 'assembled', 'rigged up', 'jury rigged'
        ]
        diy_score = sum(text.count(keyword) for keyword in diy_keywords)
        idea.diy_score = min(diy_score * 1.8, 4.0)  # Increased multiplier
        
        if diy_score > 0:
            idea.validation_signals.append('diy_evidence')
        
        # 4. Money Signals (Critical indicator) - More nuanced
        money_patterns = [r'\$\d+', r'pay.*\$', r'worth.*\$', r'invest.*\$']
        money_keywords = [
            'pay', 'buy', 'purchase', 'invest', 'worth paying', 'would pay', 
            'right now', 'happy to pay', 'willing to pay', 'cost', 'price'
        ]
        
        money_pattern_score = sum(len(re.findall(pattern, text)) for pattern in money_patterns)
        money_keyword_score = sum(text.count(keyword) for keyword in money_keywords)
        
        # Reduce weight for success stories with money mentions
        if idea.is_success_story:
            money_keyword_score *= 0.3  # Reduce weight for success stories
        
        idea.money_score = min((money_pattern_score * 1.5 + money_keyword_score * 0.8), 5.0)
        
        if idea.money_score > 0:
            idea.validation_signals.append('money_signal')
        
        # 5. Group Endorsement - Enhanced logic
        if idea.score > 0 and idea.num_comments > 5:
            comment_ratio = idea.num_comments / idea.score
            if comment_ratio > 0.3:  # Active discussion
                idea.endorsement_score = min(comment_ratio * 2.5, 3.0)  # Increased multiplier
                idea.validation_signals.append('group_endorsement')
        
        # Calculate overall pain score with adjusted weights
        idea.pain_score = (
            idea.specificity_score * 1.0 +      # Increased weight
            idea.frequency_score * 1.2 +        # Increased weight
            idea.diy_score * 1.5 +             # Increased weight
            idea.money_score * 1.2 +           # Reduced weight
            idea.endorsement_score * 1.0       # Increased weight
        )
        
        # Bonus for problem statements, penalty for success stories
        if idea.is_problem_statement:
            idea.pain_score *= 1.3
        elif idea.is_success_story:
            idea.pain_score *= 0.7
        
        return idea
    
    def classify_monetization_model(self, idea: StartupIdea) -> StartupIdea:
        """Step 5: Monetization Mapping"""
        content = str(idea.content) if idea.content is not None else ""
        title = str(idea.title) if idea.title is not None else ""
        text = f"{title} {content}".lower()
        
        model_scores = {}
        
        for model, config in self.MONETIZATION_MODELS.items():
            score = 0
            for signal in config['signals']:
                if signal in text:
                    score += 1
            
            if score > 0:
                model_scores[model] = score * config['weight']
        
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])
            idea.monetization_model = best_model[0]
        else:
            # Default classification
            if any(word in text for word in ['automate', 'tool', 'software']):
                idea.monetization_model = 'micro_saas'
            elif any(word in text for word in ['service', 'consulting', 'done for']):
                idea.monetization_model = 'done_for_you'
            else:
                idea.monetization_model = 'ugc_marketplace'
        
        return idea
    
    def estimate_market_size(self, idea: StartupIdea) -> StartupIdea:
        """Estimate market size based on Reddit signals"""
        indicators = {
            'high': ['everyone', 'all', 'every', 'massive', 'huge problem', 'industry-wide'],
            'medium': ['many', 'most', 'common', 'frequent', 'typical'],
            'low': ['some', 'few', 'occasional', 'niche']
        }
        
        content = str(idea.content) if idea.content is not None else ""
        title = str(idea.title) if idea.title is not None else ""
        text = f"{title} {content}".lower()
        
        for size, keywords in indicators.items():
            if any(keyword in text for keyword in keywords):
                idea.market_size_indicator = size
                break
        
        if not idea.market_size_indicator:
            # Default based on engagement
            if idea.score > 100 and idea.num_comments > 20:
                idea.market_size_indicator = 'high'
            elif idea.score > 20:
                idea.market_size_indicator = 'medium'
            else:
                idea.market_size_indicator = 'low'
        
        return idea
    
    def is_valid_post(self, post) -> bool:
        """Validate if a post is suitable for analysis"""
        try:
            # Check if post is deleted or removed
            if hasattr(post, 'removed_by_category') and post.removed_by_category:
                return False
            
            # Check if author is deleted
            if post.author is None:
                return False
            
            # Check if content is meaningful
            if not post.title or len(post.title.strip()) < 10:
                return False
            
            # Check if post is too old (optional)
            post_age_days = (time.time() - post.created_utc) / 86400
            if post_age_days > 365:  # Skip posts older than 1 year
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Error validating post: {e}")
            return False
    
    def is_duplicate_idea(self, idea: StartupIdea) -> bool:
        """Check if idea is a duplicate"""
        if idea.content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(idea.content_hash)
        return False
    
    def search_goldmine_subreddits(self, keywords: List[str], days_back: int = 60) -> List[StartupIdea]:
        """Step 1: Target the 7 Goldmine Subreddits with engagement tracking"""
        all_ideas = []
        
        for subreddit_name, config in self.GOLDMINE_SUBREDDITS.items():
            logger.info(f"Mining r/{subreddit_name} (focus: {config['focus']})")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Multiple search strategies - Enhanced for better problem detection
                search_queries = keywords + [
                    "I wish",
                    "someone should make",
                    "need a tool",
                    "frustrated with",
                    "there should be",
                    "why doesn't exist",
                    "problem with",
                    "hate using",
                    "annoyed by",
                    "difficult to",
                    "hard to",
                    "looking for solution",
                    "anyone know how",
                    "help me find"
                ]
                
                for query in search_queries:
                    try:
                        results = subreddit.search(query, limit=20, time_filter='month')
                        
                        for post in results:
                            # Validate post before processing
                            if not self.is_valid_post(post):
                                continue
                            
                            # Apply engagement velocity filter
                            velocity = self.calculate_engagement_velocity(post)
                            
                            if velocity > 0.1 and post.score > 3:  # Minimum thresholds
                                idea = StartupIdea(
                                    title=post.title,
                                    content=post.selftext,
                                    subreddit=subreddit_name,
                                    author=str(post.author) if post.author else 'deleted',
                                    score=post.score,
                                    num_comments=post.num_comments,
                                    created_utc=post.created_utc,
                                    url=f"https://reddit.com{post.permalink}",
                                    engagement_velocity=velocity
                                )
                                
                                # Skip duplicates
                                if self.is_duplicate_idea(idea):
                                    continue
                                
                                # Apply subreddit weight
                                idea.pain_score *= config['weight']
                                
                                all_ideas.append(idea)
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"Search error in r/{subreddit_name} for '{query}': {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error accessing r/{subreddit_name}: {e}")
                continue
        
        return all_ideas
    
    def apply_dark_funnel_tactics(self, keywords: List[str]) -> List[StartupIdea]:
        """Dark Funnel: Mining Deleted Threads & Private Subs"""
        dark_ideas = []
        
        # Tactic 1: Unddit.com for deleted posts
        logger.info("Applying Dark Funnel Tactic 1: Unddit.com")
        try:
            for keyword in keywords[:3]:  # Limit to avoid rate limits
                unddit_query = f"site:unddit.com/r/startups \"{keyword}\""
                # Note: In production, you'd use a web scraping service or API
                logger.info(f"Search suggestion: {unddit_query}")
        except Exception as e:
            logger.warning(f"Unddit search error: {e}")
        
        # Tactic 3: Comment Archaeology via Reddit search
        logger.info("Applying Dark Funnel Tactic 3: Comment Archaeology")
        try:
            for keyword in keywords[:2]:
                # Search for old comments with low engagement
                query = f"inurl:comments {keyword} site:reddit.com"
                logger.info(f"Comment archaeology query: {query}")
                
                # You could implement Google Custom Search API here
                # For now, we'll focus on Reddit's internal search
                
        except Exception as e:
            logger.warning(f"Comment archaeology error: {e}")
        
        return dark_ideas
    
    def validate_with_idea_funnel(self, idea: StartupIdea) -> Dict:
        """Step 4: The Idea Funnel Validation System"""
        validation_report = {
            'idea_title': idea.title,
            'pain_score': idea.pain_score,
            'validation_steps': []
        }
        
        # Step 1: Google search simulation (you'd implement actual search)
        search_query = f"{idea.title.replace(' ', '+')} tool"
        validation_report['validation_steps'].append({
            'step': 'google_search',
            'query': search_query,
            'result': 'simulation - check for existing solutions'
        })
        
        # Step 2: Market mention analysis (simulated)
        market_mentions = idea.num_comments * 2  # Simplified
        validation_report['validation_steps'].append({
            'step': 'market_mentions',
            'count': market_mentions,
            'threshold': 50
        })
        
        # Step 3: MVP readiness
        mvp_indicators = len(idea.validation_signals)
        validation_report['validation_steps'].append({
            'step': 'mvp_readiness',
            'indicators': mvp_indicators,
            'ready': mvp_indicators >= 3
        })
        
        return validation_report
    
    def mine_startup_ideas(self, keywords: List[str] = None, min_pain_score: float = 3.0, 
                          include_dark_funnel: bool = True) -> List[StartupIdea]:
        """Main mining function implementing complete methodology"""
        
        if not keywords:
            keywords = ['frustrated', 'need tool', 'wish there was', 'problem with', 'hate using', 'manual process']
        
        logger.info("Starting comprehensive Reddit idea mining...")
        
        # Step 1: Mine goldmine subreddits
        raw_ideas = self.search_goldmine_subreddits(keywords)
        logger.info(f"Found {len(raw_ideas)} raw ideas from goldmine subreddits")
        
        # Apply Dark Funnel tactics
        if include_dark_funnel:
            dark_ideas = self.apply_dark_funnel_tactics(keywords)
            raw_ideas.extend(dark_ideas)
        
        # Step 2: Apply Pain Scan Framework to all ideas
        processed_ideas = []
        for idea in raw_ideas:
            try:
                idea = self.apply_pain_scan_framework(idea)
                idea = self.classify_monetization_model(idea)
                idea = self.estimate_market_size(idea)
                
                # Enhanced filtering criteria
                if (idea.pain_score >= min_pain_score and 
                    idea.is_problem_statement and  # Prefer problem statements
                    not idea.is_success_story):    # Avoid success stories
                    processed_ideas.append(idea)
            except Exception as e:
                logger.warning(f"Error processing idea '{idea.title}': {e}")
                continue
        
        # Sort by pain score (highest potential first)
        processed_ideas.sort(key=lambda x: x.pain_score, reverse=True)
        
        self.ideas = processed_ideas
        logger.info(f"Qualified {len(processed_ideas)} high-potential startup ideas")
        
        return processed_ideas
    
    def generate_validation_report(self, top_n: int = 10) -> Dict:
        """Generate comprehensive validation report"""
        if not self.ideas:
            return {"error": "No ideas to analyze"}
        
        top_ideas = self.ideas[:top_n]
        
        try:
            report = {
                "summary": {
                    "total_ideas_found": len(self.ideas),
                    "average_pain_score": np.mean([idea.pain_score for idea in self.ideas]) if self.ideas else 0,
                    "high_potential_ideas": len([idea for idea in self.ideas if idea.pain_score > 6.0]),
                    "monetization_breakdown": dict(Counter([idea.monetization_model for idea in self.ideas])),
                    "market_size_distribution": dict(Counter([idea.market_size_indicator for idea in self.ideas])),
                    "problem_statements": len([idea for idea in self.ideas if idea.is_problem_statement]),
                    "success_stories": len([idea for idea in self.ideas if idea.is_success_story])
                },
                "top_ideas": [],
                "validation_reports": []
            }
            
            for idea in top_ideas:
                idea_data = {
                    "title": idea.title,
                    "pain_score": round(idea.pain_score, 2),
                    "subreddit": idea.subreddit,
                    "monetization_model": idea.monetization_model,
                    "market_size": idea.market_size_indicator,
                    "validation_signals": idea.validation_signals,
                    "engagement_velocity": round(idea.engagement_velocity, 3),
                    "url": idea.url,
                    "is_problem_statement": idea.is_problem_statement,
                    "is_success_story": idea.is_success_story,
                    "breakdown": {
                        "specificity": idea.specificity_score,
                        "frequency": idea.frequency_score,
                        "diy_evidence": idea.diy_score,
                        "money_signals": idea.money_score,
                        "endorsement": idea.endorsement_score
                    }
                }
                
                report["top_ideas"].append(idea_data)
                
                # Generate validation report for top ideas
                if idea.pain_score > 5.0:
                    validation = self.validate_with_idea_funnel(idea)
                    report["validation_reports"].append(validation)
            
            return report
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return {"error": f"Failed to generate report: {e}"}
    
    def export_results(self, filename: str = None, format: str = 'csv') -> str:
        """Export results in multiple formats"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"startup_ideas_{timestamp}"
        
        if format.lower() == 'csv':
            filename += '.csv'
            data = []
            for idea in self.ideas:
                try:
                    # Clean and encode content for CSV
                    content_preview = idea.content[:200] if idea.content else ""
                    content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
                    content_preview += '...' if len(idea.content or "") > 200 else ""
                    
                    data.append({
                        'title': idea.title,
                        'subreddit': idea.subreddit,
                        'pain_score': idea.pain_score,
                        'monetization_model': idea.monetization_model,
                        'market_size': idea.market_size_indicator,
                        'validation_signals': ';'.join(idea.validation_signals),
                        'engagement_velocity': idea.engagement_velocity,
                        'score': idea.score,
                        'comments': idea.num_comments,
                        'url': idea.url,
                        'content_preview': content_preview,
                        'is_problem_statement': idea.is_problem_statement,
                        'is_success_story': idea.is_success_story,
                        'specificity_score': idea.specificity_score,
                        'frequency_score': idea.frequency_score,
                        'diy_score': idea.diy_score,
                        'money_score': idea.money_score,
                        'endorsement_score': idea.endorsement_score
                    })
                except Exception as e:
                    logger.warning(f"Error processing idea for CSV export: {e}")
                    continue
            
            try:
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8')
            except Exception as e:
                logger.error(f"Error writing CSV file: {e}")
                # Fallback to manual CSV writing
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    if data:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            
        elif format.lower() == 'json':
            filename += '.json'
            try:
                report = self.generate_validation_report(len(self.ideas))
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error writing JSON file: {e}")
        
        logger.info(f"Results exported to {filename}")
        return filename

def load_existing_ideas(file_path: str) -> List[Dict]:
    """Load existing startup ideas from CSV or JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ideas = []
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            idea = row.to_dict()
            # Convert string representations back to proper types
            if 'validation_signals' in idea and isinstance(idea['validation_signals'], str):
                try:
                    idea['validation_signals'] = json.loads(idea['validation_signals'])
                except:
                    idea['validation_signals'] = []
            ideas.append(idea)
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'ideas' in data:
                ideas = data['ideas']
            else:
                ideas = data
    
    return ideas

def analyze_existing_ideas(file_path: str, model: str = "llama3.1", max_ideas: int = 10):
    """Analyze existing startup ideas using Ollama"""
    print(f"Loading ideas from: {file_path}")
    ideas = load_existing_ideas(file_path)
    
    if not ideas:
        print("No ideas found in the file.")
        return
    
    print(f"Found {len(ideas)} ideas. Analyzing top {min(max_ideas, len(ideas))}...")
    
    # Initialize Ollama analyzer
    analyzer = OllamaAnalyzer(model=model)
    
    # Check if Ollama is available
    if not analyzer._check_ollama_available():
        print("ERROR: Ollama is not running or not accessible.")
        print("Please start Ollama service: ollama serve")
        print("And pull a model: ollama pull llama3.1")
        return
    
    # Analyze ideas
    results = analyzer.analyze_ideas_batch(ideas, max_ideas)
    
    if 'error' in results:
        print(f"Analysis failed: {results['error']}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("OLLAMA AI ANALYSIS RESULTS")
    print("="*80)
    
    if results.get('fallback_used'):
        print("\n⚠️  Using fallback analysis (Ollama not available)")
        print("   Install Ollama for AI-powered analysis: https://ollama.ai/")
    
    summary = results['summary']
    if 'error' not in summary:
        print(f"\nAnalysis Summary:")
        print(f"  Ideas Analyzed: {summary['total_ideas_analyzed']}")
        print(f"  Average Business Score: {summary['average_scores']['business_potential']:.2f}/10")
        print(f"  Average Execution Score: {summary['average_scores']['execution_viability']:.2f}/10")
        print(f"  Average Monetization Score: {summary['average_scores']['monetization']:.2f}/10")
    
    recommendations = results['recommendations']
    if 'error' not in recommendations:
        print(f"\nRecommendations:")
        
        # Handle both new and fallback formats
        if 'recommendation_distribution' in recommendations:
            dist = recommendations['recommendation_distribution']
            print(f"  Pursue Aggressively: {dist.get('pursue_aggressively', 0)}")
            print(f"  Pursue Cautiously: {dist.get('pursue_cautiously', 0)}")
            print(f"  Validate Further: {dist.get('validate_further', 0)}")
            print(f"  Pass: {dist.get('pass', 0)}")
        else:
            # Fallback format - count recommendations manually
            pursue_count = len([a for a in results.get('analyses', []) 
                              if 'Pursue' in a.get('recommendation', '')])
            consider_count = len([a for a in results.get('analyses', []) 
                                if 'Consider' in a.get('recommendation', '')])
            pass_count = len([a for a in results.get('analyses', []) 
                            if 'Pass' in a.get('recommendation', '')])
            print(f"  Pursue: {pursue_count}")
            print(f"  Consider: {consider_count}")
            print(f"  Pass: {pass_count}")
        
        if 'strategic_recommendations' in recommendations and recommendations['strategic_recommendations']:
            print(f"\nStrategic Recommendations:")
            for rec in recommendations['strategic_recommendations']:
                print(f"  - {rec['type']}: {rec['description']}")
                if rec.get('action'):
                    print(f"    Action: {rec['action']}")
        
        if 'common_challenges' in recommendations and recommendations['common_challenges']:
            print(f"\nCommon Challenges:")
            for challenge in recommendations['common_challenges']:
                print(f"  • {challenge}")
        
        if 'high_opportunity_ideas' in recommendations and recommendations['high_opportunity_ideas']:
            print(f"\nHigh Opportunity Ideas:")
            for opp in recommendations['high_opportunity_ideas']:
                print(f"  • {opp['title']} (Pain Score: {opp['pain_score']})")
                if opp.get('reasoning'):
                    print(f"    Reasoning: {opp['reasoning']}")
        
        if 'next_actions' in recommendations and recommendations['next_actions']:
            print(f"\nNext Actions:")
            for action in recommendations['next_actions']:
                print(f"  • {action}")
        
        # Show monetization breakdown if available
        if 'monetization_breakdown' in recommendations:
            print(f"\nMonetization Model Breakdown:")
            for model, count in recommendations['monetization_breakdown'].items():
                print(f"  • {model}: {count} ideas")
        
        # Show subreddit insights if available
        if 'subreddit_insights' in recommendations:
            print(f"\nSubreddit Insights:")
            for subreddit, insights in recommendations['subreddit_insights'].items():
                if insights['count'] > 0:
                    print(f"  • r/{subreddit}: {insights['count']} ideas, "
                          f"avg pain score: {insights['avg_pain_score']:.1f}, "
                          f"high potential: {insights['high_potential_count']}")
    
    # Show top scoring ideas
    if 'top_scoring_ideas' in summary and summary['top_scoring_ideas']:
        print(f"\nTop Scoring Ideas:")
        for i, analysis in enumerate(summary['top_scoring_ideas'], 1):
            original_idea = analysis.get('original_idea', {})
            business = analysis.get('business_potential', {})
            recommendation = analysis.get('recommendation', 'N/A')
            key_insight = analysis.get('key_insight', 'N/A')
            
            print(f"\n{i}. {original_idea.get('title', 'N/A')}")
            print(f"   Business Score: {business.get('score', 0)}/10")
            print(f"   Recommendation: {recommendation}")
            print(f"   Key Insight: {key_insight}")
            if analysis.get('fallback'):
                print(f"   ⚠️  Fallback analysis used")
    
    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"ollama_analysis_{timestamp}.json"
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*80)
    print(f"Analysis results saved to: {analysis_file}")
    print("="*80)

def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="Reddit Startup Idea Miner with Ollama AI Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Scrape new ideas and analyze
  python main.py --analyze-only startup_ideas.csv   # Analyze existing CSV file
  python main.py --analyze-only startup_ideas.json  # Analyze existing JSON file
  python main.py --model llama3.2 --max-ideas 5    # Use specific model and limit
  python main.py --no-scrape                        # Skip scraping, analyze latest files
        """
    )
    
    parser.add_argument(
        '--analyze-only', 
        type=str, 
        help='Analyze existing file (CSV or JSON) without scraping'
    )
    parser.add_argument(
        '--model', 
        default='llama3.2', 
        help='Ollama model to use for analysis (default: llama3.2)'
    )
    parser.add_argument(
        '--max-ideas', 
        type=int, 
        default=10, 
        help='Maximum number of ideas to analyze (default: 10)'
    )
    parser.add_argument(
        '--no-scrape', 
        action='store_true', 
        help='Skip scraping, analyze the most recent output files'
    )
    
    args = parser.parse_args()
    
    # If analyze-only mode
    if args.analyze_only:
        try:
            analyze_existing_ideas(args.analyze_only, args.model, args.max_ideas)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Available files:")
            for file in os.listdir('.'):
                if file.endswith(('.csv', '.json')) and 'startup_ideas' in file:
                    print(f"  {file}")
        except Exception as e:
            print(f"Analysis failed: {e}")
        return
    
    # If no-scrape mode, find latest files
    if args.no_scrape:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'startup_ideas' in f]
        json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'startup_ideas' in f]
        
        if not csv_files and not json_files:
            print("No startup idea files found. Run scraping first.")
            return
        
        # Use the most recent file
        latest_file = None
        if csv_files:
            latest_file = max(csv_files, key=lambda f: os.path.getmtime(f))
        elif json_files:
            latest_file = max(json_files, key=lambda f: os.path.getmtime(f))
        
        print(f"Analyzing latest file: {latest_file}")
        try:
            analyze_existing_ideas(latest_file, args.model, args.max_ideas)
        except Exception as e:
            print(f"Analysis failed: {e}")
        return
    
    # Default mode: scrape and analyze
    print("Starting Reddit idea mining and analysis...")
    
    # Reddit API credentials - can be set via environment variables or directly
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Uvultktj9Y2XXBEB4Romdw")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "lySCb6ERpgChi2WIk_re_HG43UvRyg")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT", "StartupIdeaMiner/2026 by No-Celebration6115")
    
    # Check if credentials are provided
    if CLIENT_ID == "your_client_id_here" or CLIENT_SECRET == "your_client_secret_here":
        print("ERROR: Please update the Reddit API credentials in main.py")
        print("Get them from: https://www.reddit.com/prefs/apps/")
        print("Update CLIENT_ID and CLIENT_SECRET variables")
        sys.exit(1)
    
    try:
        # Initialize the miner
        miner = RedditIdeaMiner(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
        # Custom search configuration - Enhanced keywords for better problem detection
        search_keywords = [
            'frustrated with',
            'need a tool',
            'wish there was',
            'manually doing',
            'hate using',
            'time consuming',
            'no good solution',
            'problem with',
            'annoyed by',
            'difficult to',
            'hard to',
            'looking for solution',
            'anyone know how',
            'help me find'
        ]
        
        # Mine ideas using complete methodology
        ideas = miner.mine_startup_ideas(
            keywords=search_keywords,
            min_pain_score=4.0,
            include_dark_funnel=True
        )
        
        if not ideas:
            print("No ideas found. Try adjusting search keywords or pain score threshold.")
            return
        
        # Generate comprehensive report
        report = miner.generate_validation_report(top_n=20)
        
        # Export in multiple formats
        csv_file = miner.export_results(format='csv')
        json_file = miner.export_results(format='json')
        
        # Display top findings
        print("\n" + "="*80)
        print("ULTIMATE REDDIT IDEA SOURCING RESULTS")
        print("="*80)
        
        print(f"\nTotal Ideas Found: {report['summary']['total_ideas_found']}")
        print(f"High-Potential Ideas: {report['summary']['high_potential_ideas']}")
        print(f"Average Pain Score: {report['summary']['average_pain_score']:.2f}")
        print(f"Problem Statements: {report['summary']['problem_statements']}")
        print(f"Success Stories: {report['summary']['success_stories']}")
        
        print(f"\nMonetization Model Distribution:")
        for model, count in report['summary']['monetization_breakdown'].items():
            print(f"  {model}: {count}")
        
        print(f"\nTOP 10 STARTUP IDEAS:")
        print("-" * 80)
        
        for i, idea in enumerate(report['top_ideas'][:10], 1):
            print(f"\n{i}. {idea['title']}")
            print(f"   Pain Score: {idea['pain_score']} | Model: {idea['monetization_model']}")
            print(f"   Subreddit: r/{idea['subreddit']} | Market: {idea['market_size']}")
            print(f"   Type: {'Problem' if idea['is_problem_statement'] else 'Success Story' if idea['is_success_story'] else 'Other'}")
            print(f"   Signals: {', '.join(idea['validation_signals'])}")
            print(f"   URL: {idea['url']}")
        
        print(f"\n" + "="*80)
        print(f"Results exported to:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")
        print("="*80)
        
        # Now analyze with Ollama
        print(f"\nStarting Ollama AI analysis of top {args.max_ideas} ideas...")
        try:
            analyze_existing_ideas(csv_file, args.model, args.max_ideas)
        except Exception as e:
            print(f"Ollama analysis failed: {e}")
            print("You can run analysis later with:")
            print(f"python main.py --analyze-only {csv_file}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have valid Reddit API credentials.")
        print("Get them from: https://www.reddit.com/prefs/apps/")
    except Exception as e:
        logger.error(f"Mining failed: {e}")
        print(f"Error: {e}")
        print("\nPlease check your internet connection and Reddit API credentials.")

if __name__ == "__main__":
    main()