### **Knowledge Skeleton: ROC, AUC, and Classification Metrics**

**Part 1: The Core Concept (Theoretical Foundations)**

At its heart, classification evaluation is about one thing: quantifying the quality of a model's predictions against the true, known labels. Simple accuracy (Correct Predictions / Total Predictions) is often a poor and misleading metric, especially when dealing with **imbalanced datasets** (e.g., fraud detection, where 99.9% of transactions are not fraudulent).

To get a true picture, we use a **Confusion Matrix**, which is the foundation for almost all other classification metrics.

|                    | **Predicted Positive (1)** | **Predicted Negative (0)** |
| ------------------ | -------------------------- | -------------------------- |
| **Actual Positive (1)** | True Positive (TP)         | False Negative (FN)        |
| **Actual Negative (0)** | False Positive (FP)        | True Negative (TN)         |

From this matrix, we derive two critical rates:

1.  **True Positive Rate (TPR)**, also known as **Recall** or **Sensitivity**. This measures: "Of all the actual positives, how many did we correctly identify?"
    $TPR = Recall = \frac{TP}{TP + FN}$
    A high TPR means the model is good at finding the positive cases.

2.  **False Positive Rate (FPR)**. This measures: "Of all the actual negatives, how many did we incorrectly label as positive?"
    $FPR = \frac{FP}{FP + TN}$
    A low FPR means the model is good at not raising false alarms.

**The ROC Curve and AUC**

Most classification models (like Logistic Regression) don't just output a class (0 or 1). They output a *probability* (e.g., 0.78). We then use a **threshold** (default is often 0.5) to convert this probability into a class label. If probability > threshold, predict 1; otherwise, predict 0.

The key insight is that the choice of this threshold creates a trade-off between TPR and FPR.
*   A very **low threshold** (e.g., 0.1) will catch more true positives (high TPR) but will also incorrectly classify more negatives as positives (high FPR).
*   A very **high threshold** (e.g., 0.9) will be very selective, leading to fewer false positives (low FPR) but at the cost of missing more true positives (low TPR).

The **Receiver Operating Characteristic (ROC) Curve** is a graph that visualizes this trade-off. It plots the **TPR (y-axis) vs. the FPR (x-axis) at all possible threshold settings.**

The **Area Under the ROC Curve (AUC)** summarizes this entire curve into a single number.
*   **Intuitive Meaning:** The AUC represents the probability that a randomly chosen positive instance will be ranked higher (given a higher probability score) by the model than a randomly chosen negative instance.
*   **Scale:**
    *   **1.0:** Perfect classifier.
    *   **0.5:** No better than random guessing (this is the diagonal line on the ROC plot).
    *   **< 0.5:** Worse than random (the model's predictions are inverted).

**Why it matters:** AUC is a powerful, aggregate measure of a model's performance that is **independent of the classification threshold.** This makes it excellent for comparing different models' overall ranking ability.

---

**Part 2: The Interview Gauntlet (Theoretical Questions)**

**Conceptual Understanding:**

1.  What is a confusion matrix? Define True Positive, False Positive, True Negative, and False Negative.
2.  What are the axes of an ROC curve? What does each axis represent?
3.  Explain, in simple terms, what an AUC score of 0.85 means.
4.  What is the difference between Precision and Recall (TPR)? When is one more important than the other?
    *   **Precision:** $\frac{TP}{TP + FP}$ (Of all the things we predicted as positive, how many were actually positive?)

**Intuition & Trade-offs:**

5.  Why is accuracy often a poor metric for classification problems? Provide an example.
6.  You are building a model to detect a rare, life-threatening disease. Would you optimize for a higher TPR or a lower FPR? What does this mean for your choice of threshold?
7.  How do you extend the concept of an ROC curve and AUC to a multi-class classification problem (e.g., with classes A, B, and C)?
8.  A junior data scientist tells you their model has an AUC of 0.95. Is this model definitely "good"? What other questions would you ask?

**Troubleshooting & Edge Cases:**

9.  What does an ROC curve that hugs the top-left corner signify? What about a curve that is a straight diagonal line?
10. Can two models have the same AUC score but be different in their practical usefulness? How?
11. When would you prefer to use a Precision-Recall (PR) curve instead of an ROC curve? (Hint: Think about extreme class imbalance).
12. Your model has an AUC of 0.3. What's likely happening, and what is a simple first step you could take to fix it?

---

**Part 3: The Practical Application (Code & Implementation)**

In Python, `scikit-learn` is the go-to library for evaluation. The workflow is straightforward:

1.  **Train your model:** Fit a classifier (e.g., `LogisticRegression`, `RandomForestClassifier`) on your training data.
2.  **Get Probabilities:** Crucially, for ROC/AUC, you need the predicted probabilities for the positive class, not just the final predicted labels. You get this using the `model.predict_proba(X_test)` method. This method returns an array of shape `(n_samples, n_classes)`. For binary classification, you'll want the probabilities for class '1', which is typically the second column (`[:, 1]`).
3.  **Calculate Metrics:** The `sklearn.metrics` module has everything you need:
    *   `confusion_matrix(y_true, y_pred)`: Generates the confusion matrix. Requires hard predictions from `model.predict()`.
    *   `classification_report(y_true, y_pred)`: A text report showing precision, recall, and f1-score for each class. Also requires hard predictions.
    *   `roc_auc_score(y_true, y_scores)`: Calculates the AUC directly. Requires the true labels and the probability scores from `predict_proba`.
    *   `roc_curve(y_true, y_scores)`: Returns the FPRs, TPRs, and the thresholds used to calculate them, which you can then use to plot the curve with a library like `matplotlib`.

**Multi-Class Extension (One-vs-Rest):**

For multi-class problems, the strategy is typically **One-vs-Rest (OvR)**. To calculate the AUC for Class 'A', you treat all 'A' samples as positive and all other samples ('B', 'C', etc.) as negative. You repeat this for each class. `scikit-learn`'s `roc_auc_score` handles this automatically if you provide the right parameters (`multi_class='ovr'`, `average='macro'`).

---

**Part 4: The Code Challenge (Practical Questions)**

**Scenario:** You are a data scientist at a fintech company. You have built a logistic regression model to predict whether a customer will default on a loan. The target variable is `default` (1 for default, 0 for no default).

**Task:** Write the Python code to evaluate your trained model. Specifically:
1.  Calculate the confusion matrix and the classification report based on a default 0.5 threshold.
2.  Calculate the model's AUC score.
3.  Describe how you would find the values needed to plot the full ROC curve.

**Answer:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import numpy as np

# 1. Create Sample Data
# In a real scenario, this would be loaded from a file
data = {
    'age': [25, 30, 45, 50, 35, 28, 60, 42, 33, 55],
    'income': [50000, 80000, 120000, 40000, 95000, 60000, 75000, 150000, 45000, 85000],
    'loan_amount': [10000, 5000, 25000, 8000, 30000, 15000, 10000, 50000, 5000, 20000],
    'default': [0, 0, 0, 1, 0, 0, 1, 0, 1, 0] # Imbalanced: 3 defaults, 7 non-defaults
}
df = pd.DataFrame(data)

X = df[['age', 'income', 'loan_amount']]
y = df['default']

# 2. Split data and train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Make predictions and get probabilities
# Hard predictions for confusion matrix & classification report
y_pred = model.predict(X_test)

# Probability scores for the positive class (default=1) for AUC/ROC
y_scores = model.predict_proba(X_test)[:, 1]

# --- CODE CHALLENGE SOLUTION ---

# Question 1: Calculate confusion matrix and classification report
print("--- Evaluation based on 0.5 Threshold ---")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Output interpretation:
# TN: model correctly predicted 2 non-defaults
# FP: model incorrectly predicted 0 non-defaults as defaults
# FN: model incorrectly predicted 1 default as non-default
# TP: model correctly predicted 0 defaults

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
# The report will show precision/recall for both classes (0 and 1)

# Question 2: Calculate the AUC Score
auc_score = roc_auc_score(y_test, y_scores)
print("\n--- Threshold-Independent Evaluation ---")
print(f"AUC Score: {auc_score:.4f}")
# An AUC above 0.5 suggests the model has some predictive power.

# Question 3: Get values to plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print("\n--- Data for ROC Curve Plot ---")
print("False Positive Rates (FPR):", np.round(fpr, 2))
print("True Positive Rates (TPR):", np.round(tpr, 2))
print("Thresholds:", np.round(thresholds, 2))
# You would now use `matplotlib.pyplot.plot(fpr, tpr)` to visualize the curve.
```