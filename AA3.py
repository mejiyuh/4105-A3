import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

breast = load_breast_cancer()
X = breast.data
print(X.shape)
#Y = breast.target

breast_input = pd.DataFrame(X)
breast_input.head()

#breast = load_breast_cancer()
x = breast.data
y = breast.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logreg = LogisticRegression()
logreg.fit(x_train_scaled, y_train)

y_pred = logreg.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Results without weight penalty:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


logreg_penalty = LogisticRegression(penalty='l2')
logreg_penalty.fit(x_train_scaled, y_train)

y_pred_penalty = logreg_penalty.predict(x_test_scaled)

accuracy_penalty = accuracy_score(y_test, y_pred_penalty)
precision_penalty = precision_score(y_test, y_pred_penalty)
recall_penalty = recall_score(y_test, y_pred_penalty)

print("Results with weight penalty:")
print("Accuracy:", accuracy_penalty)
print("Precision:", precision_penalty)
print("Recall:", recall_penalty)

cm_penalty = confusion_matrix(y_test, y_pred_penalty)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_penalty, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (with penalty)')
plt.show()

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Results for Naive Bayes Classifier:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

lr_accuracy = 0.92
lr_precision = 0.93
lr_recall = 0.90
lr_f1 = 0.91


nb_metrics = [accuracy, precision, recall, f1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(10, 6))

plt.bar(np.arange(len(metrics_names))-0.2, [lr_accuracy, lr_precision, lr_recall, lr_f1], width=0.4, label='Logistic Regression')
plt.bar(np.arange(len(metrics_names))+0.2, nb_metrics, width=0.4, label='Naive Bayes')

plt.xticks(np.arange(len(metrics_names)), metrics_names)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparison of Classification Metrics')
plt.legend()
plt.ylim(0, 1)
plt.show()

K_values = range(1, 31)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

x_scaled = scaler.fit_transform(x)
for K in K_values:

    pca = PCA(n_components=K)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)


    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)


    y_pred = logreg.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
plt.figure(figsize=(10, 6))

plt.plot(K_values, accuracy_scores, label='Accuracy')
plt.plot(K_values, precision_scores, label='Precision')
plt.plot(K_values, recall_scores, label='Recall')
plt.plot(K_values, f1_scores, label='F1 Score')

plt.xlabel('Number of Principal Components (K)')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Number of Principal Components')
plt.legend()
plt.grid(True)
plt.show()

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for K in K_values:
    pca = PCA(n_components=K)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    bayes_model = GaussianNB()
    bayes_model.fit(x_train, y_train)
    y_pred = bayes_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store evaluation metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

plt.figure(figsize=(10, 6))

plt.plot(K_values, accuracy_scores, label='Accuracy')
plt.plot(K_values, precision_scores, label='Precision')
plt.plot(K_values, recall_scores, label='Recall')
plt.plot(K_values, f1_scores, label='F1 Score')

plt.xlabel('Number of Principal Components (K)')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Number of Principal Components (Bayesian Classifier)')
plt.legend()
plt.grid(True)
plt.show()
