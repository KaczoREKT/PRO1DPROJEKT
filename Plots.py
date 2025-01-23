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
        plt.savefig('PlotsJPG/accuracies.png')
        plt.show()

    def plot_metrics(self, accuracies, classification_reports):
        x = np.arange(len(accuracies))
        precision = []
        recall = []
        f1 = []

        # Iterujemy przez raporty klasyfikacji dla każdego modelu
        for report in classification_reports.values():
            # Obliczamy średnią precyzję, recall i f1-score dla bieżącego modelu
            precision.append(np.mean([v['precision'] for k, v in report.items() if isinstance(v, dict)]))
            recall.append(np.mean([v['recall'] for k, v in report.items() if isinstance(v, dict)]))
            f1.append(np.mean([v['f1-score'] for k, v in report.items() if isinstance(v, dict)]))

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
        plt.savefig('PlotsJPG/metrics.png')
        plt.show()

    def plot_confusion_matrix(self, model, X_test, y_test, title):
        y_pred = model.predict(X_test).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Class 0', 'Class 1'], cmap='Blues')
        plt.title(title)
        plt.savefig(f'PlotsJPG/{title}.png')
        plt.show()

    def plot_roc_curve(self, models, X_test, y_test):
        plt.figure(figsize=(10, 6))
        for name, model in models.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_pred_proba = model.predict(X_test).ravel()
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend()
        plt.savefig('PlotsJPG/roc_curve.png')
        plt.show()