# 📘 Permutation Feature Importance (PFI) and Stability in Explainable AI

## 1. Introduction

**Permutation Feature Importance (PFI)** is a model-agnostic method for quantifying the importance of individual input features to a predictive model. It is widely used in explainable AI (xAI) due to its simplicity and interpretability.

This document provides:
- A technical overview of PFI
- A discussion on the **stability** of PFI
- **Theoretical guarantees** and **formal theorems**
- Practical recommendations

---

## 2. What is Permutation Feature Importance?

### 📌 Definition

Given a trained model \( f \), a dataset \( D = \{(x_i, y_i)\}_{i=1}^n \), and a performance metric \( \mathcal{M} \), the PFI of feature \( j \) is defined as:

1. Compute baseline performance:

\[
\text{Perf}_\text{baseline} = \mathcal{M}(f, D)
\]

2. Randomly permute feature \( X_j \), creating \( D^{(j)} \)

3. Compute performance with shuffled feature:

\[
\text{Perf}_j = \mathcal{M}(f, D^{(j)})
\]

4. Compute importance:

\[
\text{PFI}_j = \text{Perf}_\text{baseline} - \text{Perf}_j
\]

---

## 3. Interpretation

- **High PFI** ⟹ feature is important (shuffling degrades performance).
- **Low PFI** ⟹ feature is not important.
- **Negative PFI** ⟹ model may be relying on spurious or misleading signals.

---

## 4. Stability in Explainable AI

### ✅ Why Stability Matters

- Ensures **trust** and **reproducibility** of explanations.
- Affects model **transparency**, especially in regulated domains.
- Helps filter out noise from meaningful signal.

### ❗ Factors Affecting PFI Stability

- **Feature correlation**
- **Sample size and variance**
- **Model class sensitivity**
- **Evaluation metric noise**
- **Number of permutations**

---

## 5. Theoretical Results

### 🔹 Theorem 1: Consistency of PFI

> **Statement**  
Let \( f \to f^* \) be a consistent estimator of the true function and \( \mathcal{M} \) a consistent metric. Then as \( n \to \infty \), the estimated PFI converges to a quantity proportional to the feature’s conditional influence on the target.

---

### 🔹 Theorem 2: Zero Importance for Independent Features

> **Statement**  
If \( X_j \) is independent of both \( Y \) and all other features \( X_{-j} \), then:

\[
\text{PFI}_j = 0
\]

> **Implication**  
PFI captures only informative features under perfect model learning.

---

### 🔹 Theorem 3: Instability with Collinear Features

> **Statement**  
If features \( X_j \) and \( X_k \) are highly collinear:

\[
\text{PFI}_j + \text{PFI}_k \approx \text{PFI}_{\{j,k\}}
\]

> **Implication**  
PFI values become unstable and attribution is ambiguous for collinear features.

---

### 🔹 Lemma: Boundedness of PFI

Let \( \mathcal{M} \in [0, M] \). Then:

\[
0 \leq \text{PFI}_j \leq M
\]

> Used for validation and normalization of PFI values.

---

### 🔹 Proposition: Expectation Over Permutations

\[
\text{PFI}_j = \mathbb{E}_\pi[\mathcal{M}(f, D) - \mathcal{M}(f, D^{\pi(j)})]
\]

Where \( \pi \) is a random permutation of feature \( X_j \). Justifies averaging over multiple runs.

---

### 🔹 Theorem: Asymptotic Normality

If PFI is estimated via the average over \( K \) random permutations:

\[
\sqrt{K} \left( \text{PFI}_j - \mathbb{E}[\text{PFI}_j] \right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
\]

> Enables confidence intervals and hypothesis testing.

---

## 6. PFI Stability Index

Define the **Stability Index** for feature \( j \) over \( B \) bootstrapped samples:

\[
\text{Stab}_j = 1 - \frac{\sigma_j}{\bar{\text{PFI}}_j + \delta}
\]

Where:
- \( \sigma_j \): standard deviation of PFI values
- \( \bar{\text{PFI}}_j \): mean PFI
- \( \delta \): small constant (e.g., \

