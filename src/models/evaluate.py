from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def print_evaluation(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def evaluate_and_print(y_true, y_pred):
    results = evaluate_model(y_true, y_pred)

    print("\n Model Performance:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print_evaluation(y_true, y_pred)

    return results