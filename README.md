# Reddit Startup Idea Miner

A comprehensive tool for mining startup ideas from Reddit using advanced AI analysis. This tool implements Marshall Hargrave's 2026 methodology for discovering $100k+ startup opportunities.

## 🚀 Features

- **Reddit API Integration**: Scrapes 7 goldmine subreddits for startup ideas
- **AI-Powered Analysis**: Uses Ollama for intelligent idea evaluation
- **Pain Score Framework**: 5-point pain scan methodology
- **Engagement Velocity Tracking**: Measures community interest
- **Monetization Classification**: Categorizes business models
- **Output Organization**: All results saved to `output/` directory
- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Testing**: 20 passing tests for all scoring formulas

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nmarchand73/reddit-startup-idea-miner.git
   cd reddit-startup-idea-miner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Reddit API credentials**
   - Go to https://www.reddit.com/prefs/apps
   - Create a new app (script type)
   - Use `http://localhost:8080` as the redirect URI
   - Update credentials in `constants.py` or use environment variables

4. **Install Ollama** (for AI analysis)
   ```bash
   # Download from https://ollama.ai/
   # Or use package manager
   ```

5. **Pull Ollama models**
   ```bash
   ollama serve
   ollama pull llama3.2
   # or other models like phi4-mini, deepseek-r1
   ```

## 🎯 Usage

### Basic Usage

```bash
# Scrape new ideas and analyze
python main.py

# Analyze existing file
python main.py --analyze-only output/startup_ideas_YYYYMMDD_HHMMSS.csv

# Use specific model and limit analysis
python main.py --model llama3.2 --max-ideas 5

# Skip scraping, analyze latest files
python main.py --no-scrape
```

### Command Line Options

- `--analyze-only <file>`: Analyze existing CSV/JSON file without scraping
- `--model <model>`: Specify Ollama model (default: llama3.2)
- `--max-ideas <number>`: Limit number of ideas to analyze (default: 10)
- `--no-scrape`: Skip scraping, analyze most recent output files

## 📊 Output Structure

All results are saved to the `output/` directory:

```
output/
├── startup_ideas_YYYYMMDD_HHMMSS.csv     # Raw scraped ideas
├── startup_ideas_YYYYMMDD_HHMMSS.json    # Detailed validation report
└── ollama_analysis_YYYYMMDD_HHMMSS.json  # AI analysis results
```

## 🤖 Ollama Analysis Features

The AI analysis provides:

- **Business Potential Scoring**: Market size, demand urgency, competitive landscape
- **Execution Viability**: Technical feasibility, resource requirements, time to market
- **Monetization Strength**: Revenue model clarity, pricing power, scalability
- **Risk Assessment**: Key challenges and mitigation strategies
- **Strategic Recommendations**: Actionable next steps and insights
- **Recommendation Distribution**: Pursue Aggressively/Cautiously/Validate Further/Pass

## 🧪 Testing

The project includes a comprehensive test suite with 20 passing tests:

```bash
# Run all tests
python test/run_tests.py

# Run specific test
python test/run_tests.py test_engagement_velocity_formula

# Run individual test file
python -m unittest test.test_scoring_formulas -v
```

### Test Coverage
- ✅ **Engagement Velocity Formula**: 4 test cases
- ✅ **Pain Score Formula**: 5 test cases  
- ✅ **Endorsement Scoring**: 4 test cases
- ✅ **Monetization Classification**: 4 test cases
- ✅ **Fallback Analysis Scoring**: 3 test cases
- ✅ **Integration Tests**: Complete pipeline testing

## ⚙️ Configuration

### Reddit API Credentials

Update `constants.py` or set environment variables:
```python
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
USER_AGENT = "YourApp/1.0 by YourUsername"
```

### Ollama Configuration

The system automatically detects available models and falls back gracefully if the specified model isn't available.

## 🏗️ Architecture

- **`main.py`**: Entry point and command-line interface
- **`constants.py`**: Configuration and constants
- **`startup_idea.py`**: StartupIdea dataclass and methods
- **`reddit_idea_miner.py`**: RedditIdeaMiner class for scraping
- **`ollama_analyzer.py`**: OllamaAnalyzer class for AI analysis
- **`utils.py`**: Utility functions for file operations
- **`test/`**: Comprehensive test suite

## 📈 Methodology

This tool implements the 2026 startup idea mining methodology:

1. **7 Goldmine Subreddits**: Targeted scraping from specific communities
2. **Engagement Velocity**: Custom metric for post engagement tracking
3. **5-Point Pain Scan**: Specificity, frequency, DIY evidence, money signals, group endorsement
4. **Monetization Mapping**: Micro SaaS, UGC Marketplace, Done-for-You classification
5. **Validation System**: Idea funnel validation with multiple signals

## 🔧 Troubleshooting

### Ollama Issues
- Ensure Ollama service is running: `ollama serve`
- Check available models: `ollama list`
- Pull required models: `ollama pull llama3.2`

### Reddit API Issues
- Verify credentials in `constants.py`
- Check Reddit app configuration
- Ensure proper redirect URI is set

### Output Directory
- The system automatically creates the `output/` directory
- All files are saved with timestamps for easy tracking
- Check the directory for analysis results after running

### Test Issues
- Run tests to verify formulas: `python test/run_tests.py`
- Check test documentation: `test/README.md`
- All 20 tests should pass for production use

## 📚 Documentation

- **Main README**: This file
- **Test Documentation**: `test/README.md`
- **GitHub Repository**: https://github.com/nmarchand73/reddit-startup-idea-miner

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test/run_tests.py`
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

---

**Repository**: https://github.com/nmarchand73/reddit-startup-idea-miner  
**Status**: ✅ All tests passing (20/20)  
**Last Updated**: 2025-08-04 