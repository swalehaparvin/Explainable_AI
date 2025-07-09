# Model Explainability Techniques 🔍

A comprehensive implementation of model-agnostic explainability methods including SHAP, LIME, and Feature Importance techniques for various data types (text, tabular, and image).

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Techniques Covered](#techniques-covered)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository demonstrates practical implementations of explainable AI techniques that help understand how machine learning models make decisions. These techniques are crucial for:
- Building trust in AI systems
- Debugging model behavior
- Meeting regulatory requirements
- Improving model performance

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-explainability.git
cd model-explainability

# Install required packages
pip install -r requirements.txt
```

### Requirements
```txt
shap>=0.41.0
lime>=0.2.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

## 🛠️ Techniques Covered

### 1. **Feature Importance**
Understanding which features matter most to your model's predictions.

#### Permutation Importance
- Works with any model type
- Measures importance by shuffling features and observing performance drop
- Captures feature interactions

#### Tree-based Feature Importance
- Built-in importance for tree models (Random Forest, XGBoost)
- Fast computation
- Split-based or gain-based metrics

### 2. **Model-Agnostic Explainability**

#### SHAP (SHapley Additive exPlanations)
- Unified framework based on game theory
- Provides both local and global explanations
- Consistent and theoretically sound

**Supported Explainers:**
- `TreeExplainer`: Optimized for tree-based models
- `LinearExplainer`: For linear models
- `KernelExplainer`: Works with any model (slower)
- `DeepExplainer`: For neural networks

#### Global Interpretability Tools
- Summary plots showing feature importance and effects
- Dependence plots revealing feature interactions
- Feature interaction analysis

### 3. **Local Explainability**

#### LIME (Local Interpretable Model-agnostic Explanations)
Explains individual predictions by approximating the model locally with interpretable models.

**Supported Data Types:**
- **Tabular**: Numerical and categorical features
- **Text**: Word-level importance for NLP models
- **Images**: Superpixel-based explanations

#### SHAP Local Explanations
- Waterfall plots for single predictions
- Force plots showing feature contributions
- Decision plots tracking prediction paths

## 💻 Quick Start

### Example 1: SHAP with XGBoost
```python
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
```

### Example 2: LIME for Text Classification
```python
from lime.lime_text import LimeTextExplainer

# Initialize explainer
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Explain a prediction
exp = explainer.explain_instance(
    text_instance, 
    model.predict_proba, 
    num_features=10
)

# Display explanation
exp.show_in_notebook()
```

### Example 3: Permutation Importance
```python
from sklearn.inspection import permutation_importance

# Calculate importance
perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=10
)

# Plot results
feature_names = X_test.columns
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SHAP contributors for the amazing framework
- LIME authors for pioneering local explanations
- scikit-learn team for permutation importance implementation
- The broader XAI community for continuous improvements

---

**Questions?** Open an issue or reach out on [LinkedIn](your-linkedin-url)

⭐ Star this repository if you find it helpful!
