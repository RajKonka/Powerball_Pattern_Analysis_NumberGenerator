# 🎯 Powerball Structural Emulator + Multi-Model Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational analytics and simulation tool for analyzing historical Powerball lottery data using multiple statistical models and structural pattern analysis.

---

## ⚠️ Important Disclaimer

**This application DOES NOT predict future lottery draws.**

- Lottery draws are completely random
- This tool is for **educational purposes only**
- Past patterns do not influence future outcomes
- This is a demonstration of data analysis and simulation techniques
- **Do not use this for gambling decisions**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models Explained](#models-explained)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This Streamlit application analyzes historical Powerball lottery data from the New York State Lottery API and implements multiple number generation models to study structural patterns in lottery draws. The app provides:

- **Historical Data Analysis**: Frequency analysis, hot/cold numbers, and pattern distributions
- **Multiple Generation Models**: Six different approaches to number generation
- **Comparative Analysis**: Side-by-side comparison of model outputs vs. real data
- **Advanced Structural Emulation**: GSE (Generative Structural Emulator) model that mimics real draw patterns

---

## ✨ Features

### 📊 Analysis Tab
- Hot and cold number tracking
- Powerball frequency distribution
- Pattern analysis (odd/even ratio, high/low splits)
- Sum and range distributions
- Gap analysis between consecutive numbers

### 🎟️ Generator Tab
- Generate lottery tickets using selected model
- Adjustable ticket quantity (1-50)
- CSV export functionality
- Real-time generation

### 🧪 Compare Models Tab
- Compare all six models side-by-side
- Mean metrics comparison
- Distribution alignment analysis
- Visual comparison charts

### 🧬 Model Lab Tab
- Deep dive into GSE Emulator performance
- Detailed structural comparison
- Statistical validation against historical data

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for API access)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/RajKonka/Powerball_Pattern_Analysis_NumberGenerator.git
   cd Powerball_Pattern_Analysis_NumberGenerator
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app should automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

---

## 💻 Usage

### Basic Workflow

1. **Adjust Settings** (Left Sidebar)
   - Set lookback window (50-5000 draws)
   - Configure hot/cold pool sizes
   - Set pattern tolerance levels
   - Choose number of tickets to generate

2. **Explore Tabs**
   - **Analysis**: View historical patterns and distributions
   - **Generator**: Create tickets using your selected model
   - **Compare Models**: See how different models stack up
   - **Model Lab**: Deep dive into the GSE Emulator

3. **Generate Tickets**
   - Select a model from the sidebar
   - Click "Generate Now" button
   - Download results as CSV if needed

### Tips for Best Results

- Use a **lookback window of 500+** for more stable patterns
- **Compare multiple models** to understand different approaches
- **Experiment with tolerances** to see how they affect generation
- Use the **Model Lab** to understand the GSE Emulator's behavior

---

## 🧮 Models Explained

### 1. **Pure Random (Baseline)** 🎲
- Simple random selection from 1-69 for whites, 1-26 for Powerball
- No pattern constraints
- Serves as control/baseline for comparison

### 2. **Hot-Heavy** 🔥
- Selects numbers from the most frequently drawn (hot) numbers
- Based on frequency analysis of lookback window
- Pool size configurable via slider

### 3. **Cold-Heavy** ❄️
- Selects numbers from the least frequently drawn (cold) numbers
- "Due number" theory approach
- Pool size configurable via slider

### 4. **Mixed (Hot + Cold)** 🔄
- Combines 3 hot numbers with 2 cold numbers
- Balances frequency-based approaches
- Attempts to hedge between extremes

### 5. **Pattern-Balanced** ✅
- Samples target patterns from real draw distributions
- Matches odd/even ratio
- Matches high/low split (≤34 vs >34)
- Constrains sum and max gap within tolerances
- **Most constrained traditional model**

### 6. **GSE Emulator** 🔬 (Advanced)
- **Generative Structural Emulator**
- Samples complete pattern profiles from historical data
- Uses weighted probability distribution over numbers
- Enforces multiple structural constraints:
  - Odd/even count
  - High/low count
  - Total sum
  - Number range
  - Gap distributions
- **Most sophisticated model** - attempts to emulate real draw structure

---

## 📁 Project Structure

```
Powerball_Pattern_Analysis_NumberGenerator/
│
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── documentation/                  # (Optional) Additional docs
│   └── model_details.md
│
└── data/                          # (Optional) Cached data
    └── .gitkeep
```

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

### API Settings
- **Data Source**: New York State Lottery API
- **Endpoint**: `https://data.ny.gov/resource/d6yy-54nr.json`
- **Limit**: 5000 draws
- **Update Frequency**: API updates regularly with new draws

### Model Parameters
All configurable via sidebar:
- Lookback window: 50-5000 draws
- Hot pool size: 5-30 numbers
- Cold pool size: 5-30 numbers
- Pattern tolerances: Adjustable for both models
- Ticket generation: 1-50 tickets

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Areas for Improvement
- Additional statistical models
- Enhanced visualization options
- Performance optimizations
- Unit tests
- Historical win analysis
- More detailed documentation

---

## 📊 Technical Details

### Pattern Features Analyzed
- **Odd Count**: Number of odd values in draw
- **Low Count**: Numbers ≤34 vs >34
- **Sum**: Total of all five white balls
- **Range**: Difference between max and min
- **Max Gap**: Largest gap between consecutive numbers
- **Gap Vector**: Individual gaps (g1, g2, g3, g4)

### Data Validation
- Ensures 5 unique white ball numbers
- Validates number ranges (1-69 for whites, 1-26 for PB)
- Removes duplicate or invalid draws
- Sorts data chronologically

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **New York State Lottery** for providing open API access
- **Streamlit** for the excellent web app framework
- **NumPy & Pandas** for powerful data analysis tools

---

## 📧 Contact

**Raj Konka** - [@RajKonka](https://github.com/RajKonka)

Project Link: [https://github.com/RajKonka/Powerball_Pattern_Analysis_NumberGenerator](https://github.com/RajKonka/Powerball_Pattern_Analysis_NumberGenerator)

---

## 🎓 Educational Value

This project demonstrates:
- API integration and data fetching
- Statistical analysis and pattern recognition
- Monte Carlo simulation techniques
- Probability distribution sampling
- Data visualization with Streamlit
- Comparative model evaluation
- Structural pattern emulation

Perfect for learning:
- Python data science workflows
- Streamlit application development
- Statistical modeling concepts
- Data analysis best practices

---

**Remember**: This is an educational tool. Lottery outcomes are random and unpredictable. Play responsibly!
