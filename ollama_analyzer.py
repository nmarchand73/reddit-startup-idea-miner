#!/usr/bin/env python3
"""
OllamaAnalyzer class for AI-powered analysis of startup ideas
"""

import json
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime
import ollama
import praw
from constants import OLLAMA_CONFIG
import os
from constants import OUTPUT_DIR

logger = logging.getLogger(__name__)

class OllamaAnalyzer:
    """AI-powered analyzer using Ollama for startup idea analysis"""
    
    def __init__(self, model: str = None, base_url: str = None, reddit_client: praw.Reddit = None):
        """Initialize with Ollama model and configuration"""
        self.model = model or OLLAMA_CONFIG['default_model']
        self.base_url = base_url or OLLAMA_CONFIG['base_url']
        self.timeout = OLLAMA_CONFIG['timeout']
        self.max_retries = OLLAMA_CONFIG['max_retries']
        self.reddit_client = reddit_client
        
        # Configure Ollama client (the library handles the base URL automatically)
        logger.info(f"Ollama client configured with base URL: {self.base_url}")
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available and running"""
        try:
            # Try to list models to check connectivity
            models = ollama.list()
            logger.info(f"Ollama is available. Found {len(models['models'])} models")
            return True
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            models = ollama.list()
            # Handle Ollama Python library response structure
            if hasattr(models, 'models'):
                return [model.model for model in models.models]
            elif isinstance(models, dict) and 'models' in models:
                return [model['name'] for model in models['models']]
            elif isinstance(models, list):
                return [model['name'] for model in models]
            else:
                logger.error(f"Unexpected models response structure: {models}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def _fetch_reddit_content(self, url: str) -> str:
        """Fetch the actual Reddit post content using Reddit API"""
        if not self.reddit_client:
            logger.warning("No Reddit client available, using existing content")
            return None
            
        try:
            # Extract submission ID from URL
            # URL format: https://reddit.com/r/subreddit/comments/ID/title/
            parts = url.split('/')
            if 'comments' in parts:
                comment_index = parts.index('comments')
                if comment_index + 1 < len(parts):
                    submission_id = parts[comment_index + 1]
                    
                    # Fetch the submission
                    submission = self.reddit_client.submission(id=submission_id)
                    
                    # Get the full content
                    content = submission.selftext if submission.selftext else submission.title
                    
                    # Get comments for additional context (top 3 comments)
                    submission.comment_sort = 'top'
                    submission.comments.replace_more(limit=0)  # Remove MoreComments objects
                    
                    comments_text = ""
                    for comment in submission.comments.list()[:3]:
                        if hasattr(comment, 'body') and comment.body:
                            comments_text += f"\nComment: {comment.body[:200]}..."
                    
                    full_content = f"{content}\n{comments_text}"
                    
                    logger.info(f"Fetched Reddit content: {len(full_content)} characters")
                    return full_content
                    
        except Exception as e:
            logger.error(f"Error fetching Reddit content: {e}")
            
        return None
    
    def _create_analysis_prompt(self, idea: Dict) -> str:
        """Create a detailed analysis prompt for the startup idea"""
        pain_score = idea.get('pain_score', 0)
        engagement_velocity = idea.get('engagement_velocity', 0)
        monetization_model = idea.get('monetization_model', 'Unknown')
        subreddit = idea.get('subreddit', 'Unknown')
        url = idea.get('url', 'N/A')
        
        # Try to fetch actual Reddit content
        reddit_content = self._fetch_reddit_content(url)
        
        # Use fetched content if available, otherwise fall back to existing content
        if reddit_content:
            full_content = reddit_content
            content_source = "REDDIT API (FULL CONTENT)"
        else:
            # Get existing content as fallback
            full_content = idea.get('content', idea.get('content_preview', 'N/A'))
            content_source = "EXISTING CONTENT"
            
        if len(full_content) > 3000:
            # Truncate very long content but keep more than before
            full_content = full_content[:3000] + "... (truncated)"

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
- CONTENT SOURCE: {content_source}
- FULL CONTENT: {full_content}
- URL: {url}

Provide analysis in this concise JSON format:
{{
    "score": {{
        "business": 1-10,
        "execution": 1-10,
        "monetization": 1-10
    }},
    "recommendation": "Pursue Aggressively/Pursue Cautiously/Validate Further/Pass",
    "key_insight": "Most important business insight in 1-2 sentences",
    "content_summary": "Brief 2-3 sentence summary of the main problem and context",
    "unique_selling_proposition": "Rewrite this idea as a clear, compelling USP that articulates what the business opportunity really is - who it's for, what problem it solves, and why it's unique",
    "next_steps": ["action 1", "action 2", "action 3"],
    "risks": ["risk 1", "risk 2"],
    "market_size": "small/medium/large",
    "time_to_market": "weeks/months/years"
}}

Guidelines:
- Pain scores 20+ indicate exceptional market validation
- Be specific about market opportunities and execution challenges
- Provide actionable next steps, not generic advice
- Keep insights concise and actionable
- Focus on the most critical information only
- Content summary should capture the core problem and context
- Unique selling proposition should be compelling and clearly differentiate the opportunity
"""
    
    def _parse_analysis_result(self, response: str) -> Dict:
        """Parse the analysis response from Ollama"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                
                # Clean up common JSON formatting issues
                logger.debug(f"Original JSON: {json_str}")
                json_str = self._clean_json_string(json_str)
                logger.debug(f"Cleaned JSON: {json_str}")
                
                return json.loads(json_str)
            else:
                logger.warning("Could not find JSON in response, using fallback")
                return self._fallback_analysis({})
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return self._fallback_analysis({})
        except Exception as e:
            logger.error(f"Error parsing analysis result: {e}")
            return self._fallback_analysis({})
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean up common JSON formatting issues"""
        import re
        
        # Fix missing quotes around string values that span multiple lines
        # Pattern: "key": value, (where value is not quoted and contains multiple lines)
        json_str = re.sub(r'("key_insight":\s*)([^",}]*?(?:\n[^",}]*?)*?)([,}])', r'\1"\2"\3', json_str)
        json_str = re.sub(r'("content_summary":\s*)([^",}]*?(?:\n[^",}]*?)*?)([,}])', r'\1"\2"\3', json_str)
        
        # Replace newlines within quoted strings with spaces
        json_str = re.sub(r'(")([^"]*)\n([^"]*")', r'\1\2 \3', json_str)
        
        # Remove any trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _simplify_idea_data(self, idea: Dict) -> Dict:
        """Simplify idea data to reduce redundancy and fix data types"""
        # Convert validation_signals from string to list if needed
        validation_signals = idea.get('validation_signals', [])
        if isinstance(validation_signals, str):
            try:
                # Try to parse string representation of list
                import ast
                validation_signals = ast.literal_eval(validation_signals)
            except:
                validation_signals = []
        
        # Handle NaN values
        content_preview = idea.get('content_preview')
        if content_preview is None or str(content_preview).lower() == 'nan':
            content_preview = ""
        
        return {
            'id': idea.get('id'),
            'title': idea.get('title', ''),
            'subreddit': idea.get('subreddit', ''),
            'pain_score': idea.get('pain_score', 0.0),
            'monetization_model': idea.get('monetization_model', 'Unknown'),
            'engagement_velocity': idea.get('engagement_velocity', 0.0),
            'score': idea.get('score', 0),
            'url': idea.get('url', ''),
            'validation_signals': validation_signals if isinstance(validation_signals, list) else []
        }
    
    def analyze_idea(self, idea: Dict) -> Dict:
        """Analyze a single startup idea"""
        try:
            # Check if specified model is available
            available_models = self._get_available_models()
            if not available_models:
                logger.warning("No models available, using fallback analysis")
                return self._fallback_analysis(idea)
            
            # Use specified model if available, otherwise find best match
            model_to_use = self._find_best_model_match(available_models)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(idea)
            
            # Log the prompt being sent to Ollama
            logger.info(f"=== OLLAMA PROMPT FOR IDEA: {idea.get('title', 'N/A')[:50]} ===")
            logger.info(f"Model: {model_to_use}")
            logger.info(f"Prompt:\n{prompt}")
            logger.info("=" * 80)
            
            # Generate response using Ollama
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 2048
                }
            )
            
            # Log the response received from Ollama
            logger.info(f"=== OLLAMA RESPONSE ===")
            logger.info(f"Response:\n{response['response']}")
            logger.info("=" * 80)
            
            # Parse the response
            analysis_result = self._parse_analysis_result(response['response'])
            
            # Add metadata and simplified original idea
            analysis_result['model_used'] = model_to_use
            analysis_result['original_idea'] = self._simplify_idea_data(idea)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing idea with Ollama: {e}")
            return self._fallback_analysis(idea)
    
    def _find_best_model_match(self, available_models: List[str]) -> str:
        """Find the best model match from available models"""
        # First, try exact match
        if self.model in available_models:
            return self.model
        
        # Try matching without version suffix
        base_model = self.model.split(':')[0] if ':' in self.model else self.model
        for model in available_models:
            if model.startswith(base_model):
                logger.warning(f"Model '{self.model}' not found. Using '{model}' instead.")
                return model
        
        # Try matching with version suffix
        if not self.model.endswith(':latest'):
            model_with_latest = f"{self.model}:latest"
            if model_with_latest in available_models:
                logger.warning(f"Model '{self.model}' not found. Using '{model_with_latest}' instead.")
                return model_with_latest
        
        # If no match found, use the first available model
        logger.warning(f"Model '{self.model}' not found. Using '{available_models[0]}' instead.")
        return available_models[0]
    
    def _fallback_analysis(self, idea: Dict) -> Dict:
        """Provide fallback analysis when Ollama is not available"""
        pain_score = idea.get('pain_score', 0)
        monetization_model = idea.get('monetization_model', 'Unknown')
        
        # Simple scoring based on pain score and monetization model
        business_score = min(10, pain_score / 2 + 3)
        execution_score = min(10, 8 - (pain_score / 4))
        monetization_score = min(10, 6 if monetization_model != 'Unknown' else 3)
        
        # Determine recommendation based on scores
        avg_score = (business_score + execution_score + monetization_score) / 3
        if avg_score >= 7:
            recommendation = "Pursue Aggressively"
        elif avg_score >= 5:
            recommendation = "Pursue Cautiously"
        elif avg_score >= 3:
            recommendation = "Validate Further"
        else:
            recommendation = "Pass"
        
        return {
            'score': {
                'business': business_score,
                'execution': execution_score,
                'monetization': monetization_score
            },
            'recommendation': recommendation,
            'key_insight': f"Pain score of {pain_score} indicates {'strong' if pain_score >= 15 else 'moderate'} market need",
            'content_summary': f"Post discusses {idea.get('title', 'business problem')} with pain score {pain_score}",
            'unique_selling_proposition': f"A solution targeting {idea.get('subreddit', 'business users')} with pain score {pain_score}, focusing on {idea.get('monetization_model', 'business needs')}",
            'next_steps': [
                'Conduct customer interviews',
                'Build minimum viable product',
                'Test pricing strategy'
            ],
            'risks': ['Market competition', 'Execution complexity'],
            'market_size': 'medium' if pain_score >= 15 else 'small',
            'time_to_market': 'months',
            'original_idea': idea,
            'fallback': True
        }
    
    def _create_batch_summary(self, analyses: List[Dict]) -> Dict:
        """Create a summary of batch analysis results"""
        if not analyses:
            return {"error": "No analyses to summarize"}
        
        # Calculate average scores
        business_scores = [a.get('score', {}).get('business', 0) for a in analyses]
        execution_scores = [a.get('score', {}).get('execution', 0) for a in analyses]
        monetization_scores = [a.get('score', {}).get('monetization', 0) for a in analyses]
        
        # Get top scoring ideas
        top_scoring = sorted(analyses, 
                           key=lambda x: x.get('score', {}).get('business', 0), 
                           reverse=True)[:3]
        
        # Count recommendations
        recommendations = {}
        for analysis in analyses:
            rec = analysis.get('recommendation', '').lower()
            if 'pursue aggressively' in rec:
                recommendations['pursue_aggressively'] = recommendations.get('pursue_aggressively', 0) + 1
            elif 'pursue cautiously' in rec:
                recommendations['pursue_cautiously'] = recommendations.get('pursue_cautiously', 0) + 1
            elif 'validate further' in rec:
                recommendations['validate_further'] = recommendations.get('validate_further', 0) + 1
            elif 'pass' in rec:
                recommendations['pass'] = recommendations.get('pass', 0) + 1
        
        return {
            'total_ideas': len(analyses),
            'avg_scores': {
                'business': sum(business_scores) / len(business_scores) if business_scores else 0,
                'execution': sum(execution_scores) / len(execution_scores) if execution_scores else 0,
                'monetization': sum(monetization_scores) / len(monetization_scores) if monetization_scores else 0
            },
            'recommendations': recommendations,
            'top_ideas': [
                {
                    'title': analysis.get('original_idea', {}).get('title', 'N/A'),
                    'business_score': analysis.get('score', {}).get('business', 0),
                    'recommendation': analysis.get('recommendation', 'N/A'),
                    'key_insight': analysis.get('key_insight', 'N/A')
                }
                for analysis in top_scoring
            ]
        }
    
    def _create_batch_recommendations(self, analyses: List[Dict], original_ideas: List[Dict]) -> Dict:
        """Create strategic recommendations based on batch analysis"""
        if not analyses:
            return {"error": "No analyses to create recommendations from"}
        
        # Collect all next steps and risks
        all_next_steps = []
        all_risks = []
        
        for analysis in analyses:
            all_next_steps.extend(analysis.get('next_steps', []))
            all_risks.extend(analysis.get('risks', []))
        
        # Remove duplicates and limit
        unique_next_steps = list(set(all_next_steps))[:5]
        unique_risks = list(set(all_risks))[:5]
        
        # Count monetization models
        monetization_models = {}
        for idea in original_ideas:
            model = idea.get('monetization_model', 'Unknown')
            monetization_models[model] = monetization_models.get(model, 0) + 1
        
        # Subreddit insights
        subreddit_stats = {}
        for i, idea in enumerate(original_ideas):
            subreddit = idea.get('subreddit', 'Unknown')
            if subreddit not in subreddit_stats:
                subreddit_stats[subreddit] = {
                    'count': 0,
                    'avg_pain_score': 0,
                    'high_potential': 0
                }
            
            subreddit_stats[subreddit]['count'] += 1
            subreddit_stats[subreddit]['avg_pain_score'] += idea.get('pain_score', 0)
            
            if analyses[i].get('score', {}).get('business', 0) >= 7:
                subreddit_stats[subreddit]['high_potential'] += 1
        
        # Calculate averages
        for subreddit in subreddit_stats:
            if subreddit_stats[subreddit]['count'] > 0:
                subreddit_stats[subreddit]['avg_pain_score'] /= subreddit_stats[subreddit]['count']
        
        return {
            'next_steps': unique_next_steps,
            'risks': unique_risks,
            'monetization_models': monetization_models,
            'subreddit_insights': subreddit_stats
        }
    
    def analyze_ideas_batch(self, ideas: List[Dict], max_ideas: int = 10) -> Dict:
        """Analyze a batch of startup ideas"""
        if not ideas:
            return {"error": "No ideas to analyze"}
        
        # Limit the number of ideas to analyze
        ideas_to_analyze = ideas[:max_ideas]
        
        logger.info(f"Analyzing {len(ideas_to_analyze)} ideas with Ollama...")
        
        analyses = []
        fallback_used = False
        
        # Create analysis file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(OUTPUT_DIR, f"ollama_analysis_{timestamp}.json")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize the analysis file with empty structure
        initial_data = {
            'summary': {},
            'recommendations': {},
            'analyses': [],
            'fallback_used': False,
            'total_ideas_analyzed': 0,
            'analysis_start_time': timestamp
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results will be saved to: {analysis_file}")
        
        for i, idea in enumerate(ideas_to_analyze, 1):
            logger.info(f"Analyzing idea {i}/{len(ideas_to_analyze)}: {idea.get('title', 'N/A')[:50]}...")
            
            analysis = self.analyze_idea(idea)
            
            if analysis.get('fallback'):
                fallback_used = True
            
            analyses.append(analysis)
            
            # Update the analysis file after each request
            self._append_analysis_result(analysis_file, analysis, i, len(ideas_to_analyze), fallback_used)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Generate batch summary and recommendations
        summary = self._create_batch_summary(analyses)
        recommendations = self._create_batch_recommendations(analyses, ideas_to_analyze)
        
        # Update the final summary in the file
        final_data = {
            'summary': summary,
            'recommendations': recommendations,
            'analyses': analyses,
            'fallback_used': fallback_used,
            'total_ideas_analyzed': len(analyses),
            'analysis_start_time': timestamp,
            'analysis_complete_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete. Results saved to: {analysis_file}")
        
        return final_data
    
    def _append_analysis_result(self, analysis_file: str, analysis: Dict, current_index: int, total_ideas: int, fallback_used: bool):
        """Append a single analysis result to the analysis file."""
        try:
            # Read the current file
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add the new analysis result
            data['analyses'].append(analysis)
            
            # Update progress information
            data['total_ideas_analyzed'] = current_index
            data['fallback_used'] = fallback_used
            
            # Update summary and recommendations based on current analyses
            if data['analyses']:
                data['summary'] = self._create_batch_summary(data['analyses'])
                # Extract original ideas for recommendations
                original_ideas = [a.get('original_idea', {}) for a in data['analyses']]
                data['recommendations'] = self._create_batch_recommendations(data['analyses'], original_ideas)
            
            # Write back to file
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Appended analysis {current_index}/{total_ideas} to {analysis_file}")
            
        except Exception as e:
            logger.error(f"Error appending analysis result to file {analysis_file}: {e}") 