from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy: {acc:.4f}')
    print("\nClassification Report:\n", classification_report(y_test, preds))

    plt.figure(figsize=(7,6))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()