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

import argparse
import logging
import os
import glob
from datetime import datetime

from constants import CLIENT_ID, CLIENT_SECRET, USER_AGENT, OUTPUT_DIR
from reddit_idea_miner import RedditIdeaMiner
from ollama_analyzer import OllamaAnalyzer
from utils import analyze_existing_ideas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_output_file():
    """Find the most recent output file (CSV or JSON)"""
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "startup_ideas_*.csv"))
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "startup_ideas_*.json"))
    
    all_files = csv_files + json_files
    if not all_files:
        return None
    
    # Sort by modification time and return the most recent
    latest_file = max(all_files, key=os.path.getmtime)
    return latest_file

def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="Reddit Startup Idea Miner with Ollama AI Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_new.py                                    # Scrape new ideas and analyze
  python main_new.py --analyze-only startup_ideas.csv   # Analyze existing CSV file
  python main_new.py --analyze-only startup_ideas.json  # Analyze existing JSON file
  python main_new.py --model llama3.2 --max-ideas 5    # Use specific model and limit
  python main_new.py --no-scrape                        # Skip scraping, analyze latest files
        """
    )
    
    parser.add_argument(
        '--analyze-only', 
        type=str, 
        help='Analyze existing file (CSV or JSON) without scraping'
    )
    parser.add_argument(
        '--model', 
        default='llama3.2:latest', 
        help='Ollama model to use for analysis (default: llama3.2:latest)'
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
    
    # Handle analyze-only mode
    if args.analyze_only:
        try:
            analyze_existing_ideas(args.analyze_only, args.model, args.max_ideas)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Analysis failed: {e}")
        return
    
    # Handle no-scrape mode
    if args.no_scrape:
        latest_file = find_latest_output_file()
        if latest_file:
            print(f"Found latest file: {latest_file}")
            try:
                analyze_existing_ideas(latest_file, args.model, args.max_ideas)
            except Exception as e:
                print(f"Analysis failed: {e}")
        else:
            print("No output files found. Run scraping first or specify a file with --analyze-only")
        return
    
    # Default mode: scrape and analyze
    try:
        # Initialize Reddit miner
        miner = RedditIdeaMiner(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
        # Mine startup ideas
        print("Mining startup ideas from Reddit...")
        ideas = miner.mine_startup_ideas()
        
        if not ideas:
            print("No ideas found. Try different keywords or check Reddit API credentials.")
            return
        
        # Export results
        print(f"\nFound {len(ideas)} ideas. Exporting results...")
        csv_file = miner.export_results(format='csv')
        json_file = miner.export_results(format='json')
        
        print(f"Results exported to:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")
        
        # Analyze with Ollama
        print(f"\nAnalyzing top {args.max_ideas} ideas with Ollama...")
        analyzer = OllamaAnalyzer(model=args.model, reddit_client=miner.reddit)

        # Convert ideas to dict format for analysis
        ideas_dict = [idea.to_dict() for idea in ideas]
        
        # Add full content for analysis (not just preview)
        for idea_dict in ideas_dict:
            # Use full content, fallback to preview if not available
            full_content = idea_dict.get('content', '')
            if not full_content and 'content_preview' in idea_dict:
                full_content = idea_dict['content_preview']
            idea_dict['content'] = full_content
        
        results = analyzer.analyze_ideas_batch(ideas_dict, args.max_ideas)
        
        if 'error' in results:
            print(f"Analysis failed: {results['error']}")
            return
        
        # Display results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Total ideas analyzed: {summary.get('total_ideas', 0)}")
            print(f"Average scores - Business: {summary.get('avg_scores', {}).get('business', 0):.1f}, "
                  f"Execution: {summary.get('avg_scores', {}).get('execution', 0):.1f}, "
                  f"Monetization: {summary.get('avg_scores', {}).get('monetization', 0):.1f}")
            
            recommendations = summary.get('recommendations', {})
            if recommendations:
                print("\nRecommendations breakdown:")
                for rec, count in recommendations.items():
                    print(f"  {rec.replace('_', ' ').title()}: {count}")
        
        if 'top_ideas' in results.get('summary', {}):
            print(f"\nTop ideas (ranked by business score):")
            for i, idea in enumerate(results['summary']['top_ideas'], 1):
                print(f"{i}. {idea.get('title', 'N/A')[:60]}...")
                print(f"   Business Score: {idea.get('business_score', 0)}, "
                      f"Recommendation: {idea.get('recommendation', 'N/A')}")
                print(f"   Key Insight: {idea.get('key_insight', 'N/A')[:100]}...")
                print()
        
        # Display detailed analysis with USPs
        if 'analyses' in results:
            print("\n" + "=" * 80)
            print("DETAILED ANALYSIS WITH UNIQUE SELLING PROPOSITIONS")
            print("=" * 80)
            for i, analysis in enumerate(results['analyses'], 1):
                original_idea = analysis.get('original_idea', {})
                print(f"\n{i}. {original_idea.get('title', 'N/A')}")
                print(f"   Pain Score: {original_idea.get('pain_score', 0):.1f}")
                print(f"   Subreddit: r/{original_idea.get('subreddit', 'N/A')}")
                print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")
                print(f"   Business Score: {analysis.get('score', {}).get('business', 0)}")
                print(f"   USP: {analysis.get('unique_selling_proposition', 'N/A')}")
                print(f"   Key Insight: {analysis.get('key_insight', 'N/A')}")
                print(f"   Content Summary: {analysis.get('content_summary', 'N/A')}")
                print()
        
        if 'recommendations' in results:
            recs = results['recommendations']
            if 'next_steps' in recs:
                print("Key Next Steps:")
                for i, step in enumerate(recs['next_steps'][:3], 1):
                    print(f"  {i}. {step}")
                print()
            
            if 'risks' in recs:
                print("Key Risks:")
                for i, risk in enumerate(recs['risks'][:3], 1):
                    print(f"  {i}. {risk}")
                print()
        
        print("=" * 80)
        print(f"Analysis results saved to: {results.get('analysis_file', 'N/A')}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()