import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
class Plots:
    def __init__(self):
        pass

    def plot_accuracies(self, accuracies):
        models = list(accuracies.keys())
        scores = list(accuracies.values())

        plt.figure(figsize=(10, 6))
        plt.bar(models, scores, color='skyblue')
        plt.title("Model Accuracies", fontsize=16)
        plt.xlabel("Models", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.ylim(0, 1)
        for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12)
        plt.show()

    def plot_metrics(self, accuracies, classification_reports):
        x = np.arange(len(accuracies))

        precision = [np.mean([v['precision'] for k, v in report.items() if isinstance(v, dict)]) for report in
                     classification_reports.values()]
        recall = [np.mean([v['recall'] for k, v in report.items() if isinstance(v, dict)]) for report in
                  classification_reports.values()]
        f1 = [np.mean([v['f1-score'] for k, v in report.items() if isinstance(v, dict)]) for report in
              classification_reports.values()]

        width = 0.25
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', color='lightgreen')
        plt.bar(x, recall, width, label='Recall', color='skyblue')
        plt.bar(x + width, f1, width, label='F1-Score', color='salmon')

        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Scores', fontsize=14)
        plt.title('Metrics Comparison', fontsize=16)
        plt.xticks(x, accuracies.keys(), fontsize=12)
        plt.legend()
        plt.ylim(0, 1)
        plt.show()

    def plot_confusion_matrix(self, model, X_test, y_test, title):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title(title)
        plt.show()

    def plot_roc_curve(self, models, X_test, y_test):
        plt.figure(figsize=(10, 6))
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend()
        plt.show()