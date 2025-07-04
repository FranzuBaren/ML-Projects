# 📊 Permutation Feature Importance

## 🔍 What is Permutation Feature Importance?

**Permutation Feature Importance** is a **model-agnostic** technique to measure the importance of individual features in a predictive model.

- It quantifies how much the **model performance deteriorates** when the values of a feature are randomly **permuted**, thus **breaking the relationship** between that feature and the target variable.
- The greater the drop in performance, the more important the feature.

---

## ✏️ Intuition

1. **Train your model** as usual.
2. **Measure baseline performance** (e.g., accuracy, R², RMSE).
3. For each feature:
   - Shuffle its values, leaving all other features untouched.
   - Measure the new performance.
   - Compute the performance drop:

   $$
   \text{Importance} = \text{Performance}_{\text{baseline}} - \text{Performance}_{\text{shuffled}}
   $$

4. **Rank features** according to the drop in performance.

---

## ✅ Advantages

- **Model-agnostic:** Works with any model (tree-based, linear, neural networks, etc.)
- **Simple to implement.**
- Captures **feature interactions** (unlike some other importance measures).
- Reflects **actual contribution** to the predictive power.

---

## ⚠️ Limitations

- **Computationally expensive:** Requires re-evaluating the model for each feature permutation.
- Can **underestimate correlated features’ importance**. If two features are correlated, permuting one might not reduce performance much since the other still contains similar information.
- Sensitive to **the choice of performance metric**.

---

## ⚙️ Common Performance Metrics

- Regression:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - $R^2$ Score
- Classification:
  - Accuracy
  - F1 Score
  - AUC-ROC

---

## 🛠️ Example (Python - `scikit-learn`)

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Example data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Baseline accuracy
baseline_score = model.score(X_test, y_test)

# Permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Print results
for i in result.importances_mean.argsort()[::-1]:
    print(f"Feature {i}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")
```

---

## 🔑 Key Takeaways

- Permuta
