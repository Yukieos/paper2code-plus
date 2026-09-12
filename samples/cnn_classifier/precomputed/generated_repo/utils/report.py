import numpy as np

def report_results(accuracy: float, confusion_matrix: np.ndarray) -> None:
    """
    Reports the evaluation results including accuracy and confusion matrix.

    Parameters:
    accuracy (float): Accuracy of the model.
    confusion_matrix (np.ndarray): Confusion matrix for the model's predictions.
    """
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix)