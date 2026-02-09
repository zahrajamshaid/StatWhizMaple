# 📊 StatMaple - Smart Statistical & ML Toolkit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A professional, all-in-one data analysis and machine learning web application designed to impress recruiters and accelerate data science workflows.**

StatMaple is a comprehensive toolkit that combines exploratory data analysis, statistical testing, data visualization, and machine learning into one intuitive web interface. Perfect for data scientists, analysts, and anyone who wants to quickly analyze datasets and build ML models without writing code.

---

## 🌟 Features

### 📂 **Smart Data Loading**
- Upload CSV files with automatic data type detection
- Instant data quality assessment
- Missing value detection and reporting
- Sample dataset loading for testing

### 🔍 **Exploratory Data Analysis (EDA)**
- **Summary Statistics**: Mean, median, mode, std, variance, skewness, kurtosis
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
- **Missing Values Report**: Automated analysis with recommendations
- **Data Quality Score**: Overall dataset quality grading
- **Outlier Detection**: IQR and Z-score methods

### 📊 **Advanced Visualizations**
- **Histograms**: Distribution analysis with mean/median lines
- **Correlation Heatmaps**: Beautiful color-coded correlation matrices
- **Scatter Plots**: With trend lines and color coding
- **Box Plots**: Outlier detection and visualization
- **Categorical Distributions**: Bar charts and pie charts
- **Interactive Plotly Charts**: Hover data and zoom capabilities

### 📈 **Statistical Tests**
- **T-Tests**: One-sample, independent, and paired t-tests
- **Chi-Square Test**: Test of independence for categorical variables
- **ANOVA**: One-way analysis of variance
- **Correlation Tests**: Pearson and Spearman with significance testing
- **Normality Tests**: Shapiro-Wilk test for distribution analysis

### 🤖 **Machine Learning**
- **Linear Regression**: For continuous target variables
  - R², MAE, MSE, RMSE metrics
  - Residual plots and prediction visualization
  
- **Logistic Regression**: For classification tasks
  - Accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  
- **Random Forest**: Both classification and regression
  - Feature importance analysis
  - Robust to outliers and non-linear relationships
  
- **Smart Model Recommender**: AI-powered model selection
  - Automatic problem type detection (classification vs regression)
  - Data characteristic analysis
  - Model recommendations with explanations

### 🔧 **Data Preprocessing**
- Automated missing value handling (mean, median, mode, forward/backward fill)
- Categorical encoding (Label encoding, One-hot encoding)
- Feature scaling (Standard, MinMax, Robust scaling)
- Train-test splitting with stratification
- Complete ML pipeline automation

### 📋 **Model Comparison**
- Side-by-side model performance comparison
- Visual comparison charts
- Best model selection guidance

---

## 🏗️ Project Structure

```
StatWhizMaple/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
│
└── src/                       # Source modules
    ├── __init__.py
    ├── data_loader.py         # CSV loading & data inspection
    ├── eda.py                 # Exploratory data analysis
    ├── visualization.py       # Plotting and charts
    ├── stats_tests.py         # Statistical hypothesis tests
    ├── utils.py               # Data preprocessing utilities
    ├── ml_models.py           # Machine learning models
    └── recommender.py         # Smart model recommendation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/StatWhizMaple.git
cd StatWhizMaple
```

2. **Create virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1. **Upload Your Data**
- Navigate to the **📂 Data Upload** section
- Upload a CSV file or load a sample dataset
- Review data preview and overview statistics

### 2. **Explore Your Data**
- Go to **🔍 EDA** section
- View summary statistics, correlations, and missing values
- Check data quality score

### 3. **Visualize**
- Open **📊 Visualization** section
- Generate histograms, heatmaps, scatter plots, and more
- Create interactive Plotly visualizations

### 4. **Run Statistical Tests**
- Access **📈 Statistics** section
- Choose from t-tests, chi-square, ANOVA, correlation tests
- Get automated interpretations with p-values

### 5. **Train ML Models**
- Navigate to **🤖 Machine Learning** section
- Get smart model recommendations
- Configure preprocessing options
- Train and compare multiple models
- View feature importance and predictions

### 6. **Compare & Export**
- Review model comparison tables
- Select the best performing model
- Export results and visualizations

---

## 🧪 Example Workflow

```python
# 1. Load data
Upload your CSV file → View data preview

# 2. Analyze
EDA Tab → Check summary stats and correlations
Visualization Tab → Generate correlation heatmap

# 3. Test hypotheses
Statistics Tab → Run ANOVA to compare groups

# 4. Build models
ML Tab → Get recommendation → Train Random Forest
View feature importance → Compare with Linear Regression

# 5. Deploy
Select best model → Export results
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning algorithms |
| **SciPy** | Statistical tests |
| **Statsmodels** | Advanced statistics |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |
| **Plotly** | Interactive charts |

---

## 💼 Skills Demonstrated

This project showcases the following professional skills:

✅ **Python Programming**: Advanced Python with OOP principles  
✅ **Data Analysis**: Comprehensive EDA and statistical analysis  
✅ **Data Visualization**: Multiple visualization libraries and techniques  
✅ **Statistical Testing**: Hypothesis testing and interpretation  
✅ **Machine Learning**: Supervised learning (regression & classification)  
✅ **Model Evaluation**: Proper metrics and validation techniques  
✅ **Web Development**: Full-stack data science application  
✅ **Software Engineering**: Clean code, modular architecture, documentation  
✅ **User Experience**: Intuitive UI/UX design  
✅ **Project Structure**: Industry-standard organization  

---

## 📊 Sample Datasets

You can test StatMaple with these popular datasets:

1. **Iris Dataset** - Classification (flower species)
2. **Boston Housing** - Regression (house prices)
3. **Titanic** - Classification (survival prediction)
4. **Wine Quality** - Classification/Regression
5. **Diabetes** - Regression

Or use your own CSV files!

---

## 🎯 Use Cases

- **Business Analytics**: Analyze sales data, customer behavior
- **Academic Research**: Statistical analysis and hypothesis testing
- **Data Science Projects**: Quick model prototyping and comparison
- **Learning**: Understand how different algorithms work
- **Portfolio**: Demonstrate data science skills to recruiters

---

## 🔮 Future Enhancements

- [ ] Support for Excel and JSON file formats
- [ ] More ML algorithms (XGBoost, SVM, Neural Networks)
- [ ] Time series analysis and forecasting
- [ ] Automated report generation (PDF export)
- [ ] Model hyperparameter tuning interface
- [ ] Database connectivity (SQL, MongoDB)
- [ ] SHAP values for model explainability
- [ ] A/B testing framework
- [ ] Automated feature engineering

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Zahra**

- Portfolio: [Your Portfolio Link]
- LinkedIn: [Your LinkedIn]
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Inspired by the need for accessible data science tools
- Built with love for the data science community
- Special thanks to the open-source community

---

## 📸 Screenshots

### Home Page
![Home Page](screenshots/home.png)
*Clean, intuitive interface with easy navigation*

### Data Upload & Preview
![Data Upload](screenshots/upload.png)
*Instant data quality assessment and preview*

### Exploratory Data Analysis
![EDA](screenshots/eda.png)
*Comprehensive statistical summaries and correlations*

### Interactive Visualizations
![Visualizations](screenshots/viz.png)
*Beautiful, interactive charts and plots*

### Statistical Testing
![Statistics](screenshots/stats.png)
*Professional statistical analysis with interpretations*

### Machine Learning
![ML Models](screenshots/ml.png)
*Train, evaluate, and compare ML models*

---

## 🎓 Learning Resources

If you want to learn more about the concepts used in StatMaple:

- **Statistics**: [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/)
- **Data Visualization**: [Python Graph Gallery](https://www.python-graph-gallery.com/)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)

---

## 💡 Tips for Interviews

When discussing this project in interviews:

1. **Architecture**: Explain the modular design and separation of concerns
2. **Scalability**: Discuss how the code could be extended for production
3. **Testing**: Mention the built-in validation and error handling
4. **Best Practices**: Highlight clean code, documentation, and type hints
5. **User Experience**: Emphasize the intuitive UI and helpful guidance
6. **Real-world Application**: Provide examples of how this could be used in business

---

## 📞 Support

If you have any questions or need help:

- 📧 Email: your.email@example.com
- 💬 Open an issue on GitHub
- 🌐 Check the [Wiki](wiki-link) for detailed documentation

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ and ☕ by Zahra**

</div>
