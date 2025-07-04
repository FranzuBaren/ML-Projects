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

1. Compute baseline per
