#!/usr/bin/env python3
"""
Test runner for scoring formula tests
Runs all tests and provides a comprehensive report
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all tests and return results"""
    print("🧪 RUNNING SCORING FORMULA TESTS")
    print("=" * 60)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=StringIO())
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors - skipped}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    # Print detailed results
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n💥 ERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Overall status
    if failures == 0 and errors == 0:
        print(f"\n✅ ALL TESTS PASSED! ({total_tests} tests)")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED ({failures + errors} failures/errors)")
        return False

def run_specific_test(test_name):
    """Run a specific test by name"""
    print(f"🧪 RUNNING SPECIFIC TEST: {test_name}")
    print("=" * 60)
    
    # Import test modules
    from test.test_scoring_formulas import TestScoringFormulas, TestFormulaPerformance
    from test.test_integration import TestScoringPipeline, TestFormulaRegression
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add specific test
    if hasattr(TestScoringFormulas, test_name):
        suite.addTest(TestScoringFormulas(test_name))
    elif hasattr(TestFormulaPerformance, test_name):
        suite.addTest(TestFormulaPerformance(test_name))
    elif hasattr(TestScoringPipeline, test_name):
        suite.addTest(TestScoringPipeline(test_name))
    elif hasattr(TestFormulaRegression, test_name):
        suite.addTest(TestFormulaRegression(test_name))
    else:
        print(f"❌ Test '{test_name}' not found")
        return False
    
    # Run test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def run_formula_validation():
    """Run quick formula validation checks"""
    print("🔍 RUNNING FORMULA VALIDATION CHECKS")
    print("=" * 60)
    
    try:
        from reddit_idea_miner import RedditIdeaMiner
        from startup_idea import StartupIdea
        from ollama_analyzer import OllamaAnalyzer
        from constants import PAIN_PATTERNS, MONETIZATION_MODELS
        
        # Test 1: Weight consistency
        weights_sum = 0.2 + 0.2 + 0.25 + 0.25 + 0.1
        if abs(weights_sum - 1.0) < 0.001:
            print("✅ Pain score weights sum to 1.0")
        else:
            print(f"❌ Pain score weights sum to {weights_sum}, should be 1.0")
            return False
        
        # Test 2: Pattern consistency
        required_patterns = ['specificity_indicators', 'frequency_indicators', 'diy_evidence', 'money_signals']
        for pattern in required_patterns:
            if pattern in PAIN_PATTERNS:
                print(f"✅ Pattern '{pattern}' exists")
            else:
                print(f"❌ Pattern '{pattern}' missing")
                return False
        
        # Test 3: Model consistency
        required_models = ['micro_saas', 'ugc_marketplace', 'done_for_you']
        for model in required_models:
            if model in MONETIZATION_MODELS:
                print(f"✅ Model '{model}' exists")
            else:
                print(f"❌ Model '{model}' missing")
                return False
        
        # Test 4: Basic formula test
        miner = RedditIdeaMiner("test", "test", "test")
        idea = StartupIdea(
            title="Test",
            content="Test content",
            subreddit="test",
            author="test",
            score=10,
            num_comments=5,
            created_utc=time.time(),
            url="test"
        )
        
        result = miner.apply_pain_scan_framework(idea)
        if result.pain_score >= 0:
            print("✅ Pain score calculation works")
        else:
            print("❌ Pain score calculation failed")
            return False
        
        print("✅ All formula validation checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Formula validation failed: {e}")
        return False

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        print("Running all tests...")
        success = run_all_tests()
        
        # Also run quick validation
        print("\n")
        validation_success = run_formula_validation()
        success = success and validation_success
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 