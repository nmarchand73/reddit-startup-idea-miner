# Scoring Formula Tests

Comprehensive non-regression tests for all scoring formulas in the Reddit startup idea miner.

## 🚀 Quick Start

```bash
# Run all tests
python test/run_tests.py

# Run specific test
python test/run_tests.py test_engagement_velocity_formula

# Run individual test file
python -m unittest test.test_scoring_formulas -v
```

## 📋 Test Files

- **`test_scoring_formulas.py`**: Unit tests for individual formulas
- **`test_integration.py`**: Integration tests for complete pipeline
- **`run_tests.py`**: Test runner with comprehensive reporting

## 📊 Coverage

### Formula Coverage
- ✅ Engagement Velocity Formula
- ✅ Pain Score Formula (5 components)
- ✅ Endorsement Scoring
- ✅ Monetization Classification
- ✅ Fallback Analysis Scoring
- ✅ Recommendation Thresholds

### Test Categories
1. **Formula Validation**: Mathematical correctness
2. **Integration**: End-to-end pipeline testing
3. **Regression**: Formula stability over time
4. **Performance**: Time/memory efficiency

## 🧮 Mathematical Validation

### Verified Properties
- **Weight Consistency**: Pain score weights sum to 1.0
- **Bound Checking**: All scores properly bounded [0-10]
- **Non-negative Outputs**: No negative scores possible
- **Edge Case Handling**: Zero division, empty inputs
- **Deterministic Results**: Same inputs → same outputs

### Performance Benchmarks
- **Time**: < 1 second for 1000 iterations
- **Memory**: < 1KB increase per operation
- **Consistency**: Identical results for same inputs

## 🔧 Test Features

### Comprehensive Coverage
- All 5 scoring formulas thoroughly tested
- Edge cases and boundary conditions covered
- Performance benchmarks established
- Regression detection implemented

### Flexible Assertions
- Floating-point precision handled with `assertAlmostEqual()`
- Classification flexibility for monetization models
- Range checking with `assertGreater()` and `assertLess()`
- Type validation with `assertIsInstance()`

## 📈 Current Status

- **Total Tests**: 20
- **Passed**: 20 ✅
- **Failed**: 0 ❌
- **Errors**: 0 💥

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure parent directory is in Python path
2. **Mock Issues**: Tests use mocks for external dependencies
3. **Formula Changes**: Update expected values if intentionally changed
4. **Performance Issues**: Adjust time thresholds on slow systems

### Debug Mode
```bash
# Run with verbose output
python -m unittest test.test_scoring_formulas -v

# Run specific test with debug info
python test/run_tests.py test_pain_score_formula
```

## 📚 Test Naming Convention

- `test_<formula_name>_<aspect>`: Tests specific formula aspects
- `test_<pipeline_step>`: Tests pipeline integration
- `test_<edge_case>`: Tests edge cases and error conditions
- `test_<performance_metric>`: Tests performance characteristics

## 🎯 Benefits

### For Developers
- **Confidence**: Know formulas work correctly
- **Safety**: Catch regressions before deployment
- **Documentation**: Tests serve as living documentation
- **Refactoring**: Safe to modify formulas with test coverage

### For Users
- **Reliability**: Consistent and accurate scoring
- **Performance**: Efficient formula execution
- **Stability**: Formulas don't change unexpectedly
- **Quality**: Thoroughly tested mathematical operations

---

*Last Updated: 2025-08-04*
*Test Status: ✅ All 20 tests passing* 