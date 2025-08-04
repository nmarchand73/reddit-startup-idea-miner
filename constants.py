#!/usr/bin/env python3
"""
Constants and configuration for the Reddit Startup Idea Miner
"""

import re
from typing import Dict, List

# Reddit API Configuration
CLIENT_ID = "Uvultktj9Y2XXBEB4Romdw"
CLIENT_SECRET = "lySCb6ERpgChi2WIk_re_HG43UvRyg"
USER_AGENT = "StartupIdeaMiner/2026 by No-Celebration6115"

# The 7 Goldmine Subreddits (from methodology)
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
        'keywords': ['saas', 'subscription', 'monthly', 'recurring', 'software', 'app', 'tool', 'platform'],
        'patterns': [r'\$\d+/\w+', r'subscription', r'monthly fee', r'recurring revenue'],
        'signals': ['automate', 'tool', 'dashboard', 'analytics', 'integration', 'api', 'tedious task'],
        'weight': 3.0
    },
    'ugc_marketplace': {
        'keywords': ['marketplace', 'connect', 'buyers', 'sellers', 'community', 'network', 'exchange'],
        'patterns': [r'marketplace', r'connect.*with', r'buyers.*sellers'],
        'signals': ['connect', 'find', 'marketplace', 'platform', 'freelance', 'creators', 'no platform for'],
        'weight': 2.5
    },
    'done_for_you': {
        'keywords': ['service', 'consulting', 'agency', 'freelance', 'outsource', 'hire', 'professional'],
        'patterns': [r'done for you', r'hire.*to', r'professional.*service'],
        'signals': ['template', 'done for you', 'service', 'agency', 'consulting', 'pre-made'],
        'weight': 2.0
    }
}

# Success story indicators
SUCCESS_INDICATORS = [
    'i made', 'i earned', 'i built', 'i launched', 'i sold', 'i scaled',
    'success story', 'how i', 'my journey', 'from 0 to', 'reached',
    'achieved', 'hit', 'made $', 'earned $', 'revenue', 'mrr', 'arr'
]

# Problem statement indicators
PROBLEM_INDICATORS = [
    'i need', 'i want', 'i wish', 'someone should', 'there should be',
    'why doesn\'t', 'frustrated', 'annoyed', 'hate', 'sucks', 'terrible',
    'problem with', 'issue with', 'struggling', 'difficult', 'hard',
    'looking for', 'anyone know', 'help me', 'advice needed'
]

# Default search keywords for idea mining
DEFAULT_KEYWORDS = [
    'startup idea', 'business opportunity', 'pain point', 'problem', 'frustrated',
    'need help', 'looking for', 'wish there was', 'someone should make',
    'automation', 'saas', 'tool', 'app', 'platform', 'service'
]

# Validation thresholds
PAIN_SCORE_THRESHOLDS = {
    'exceptional': 20,
    'high': 15,
    'moderate': 10,
    'low': 5
}

ENGAGEMENT_VELOCITY_THRESHOLDS = {
    'high': 0.5,
    'moderate': 0.2,
    'low': 0.1
}

# File naming patterns
FILE_PATTERNS = {
    'startup_ideas': 'startup_ideas_{timestamp}.{ext}',
    'ollama_analysis': 'ollama_analysis_{timestamp}.json',
    'validation_report': 'validation_report_{timestamp}.json'
}

# Output directory configuration
OUTPUT_DIR = 'output'

# Ollama configuration
OLLAMA_CONFIG = {
    'default_model': 'llama3.2:latest',
    'base_url': 'http://localhost:11434',
    'timeout': 30,
    'max_retries': 3
}

# Analysis scoring weights
ANALYSIS_WEIGHTS = {
    'business_potential': 0.4,
    'execution_viability': 0.3,
    'monetization': 0.3
}

# Recommendation thresholds
RECOMMENDATION_THRESHOLDS = {
    'pursue_aggressively': 7.0,
    'pursue_cautiously': 5.0,
    'validate_further': 3.0,
    'pass': 0.0
} 