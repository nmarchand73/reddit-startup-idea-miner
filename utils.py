#!/usr/bin/env python3
"""
Utility functions for the Reddit Startup Idea Miner
"""

import json
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime
from ollama_analyzer import OllamaAnalyzer
from constants import OUTPUT_DIR, CLIENT_ID, CLIENT_SECRET, USER_AGENT
import praw

def load_existing_ideas(file_path: str) -> List[Dict]:
    """Load existing startup ideas from CSV or JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ideas = []
    
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to list of dictionaries
            ideas = df.to_dict('records')
            print(f"Loaded {len(ideas)} ideas from CSV file")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    elif file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ideas = json.load(f)
            print(f"Loaded {len(ideas)} ideas from JSON file")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise
    else:
        raise ValueError("Unsupported file format. Use .csv or .json files")
    
    # Sort ideas by pain_score in descending order
    ideas = sorted(ideas, key=lambda x: x.get('pain_score', 0), reverse=True)
    print(f"Sorted {len(ideas)} ideas by pain_score (highest first)")
    
    return ideas

def analyze_existing_ideas(file_path: str, model: str = "llama3.2:latest", max_ideas: int = 10):
    """Analyze existing startup ideas using Ollama"""
    print(f"Loading ideas from: {file_path}")
    ideas = load_existing_ideas(file_path)

    if not ideas:
        print("No ideas found in the specified file.")
        return

    print(f"Found {len(ideas)} ideas. Analyzing top {max_ideas}...")
    
    # Create Reddit client for fetching actual post content
    try:
        reddit_client = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        print("Reddit client created for fetching actual post content")
    except Exception as e:
        print(f"Warning: Could not create Reddit client: {e}")
        reddit_client = None
    
    analyzer = OllamaAnalyzer(model=model, reddit_client=reddit_client)
    results = analyzer.analyze_ideas_batch(ideas, max_ideas)

    if 'error' in results:
        print(f"Analysis failed: {results['error']}")
        return

    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    analysis_file = os.path.join(OUTPUT_DIR, f"ollama_analysis_{timestamp}.json")

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "="*80)
    print(f"Analysis results saved to: {analysis_file}")
    print("="*80) 