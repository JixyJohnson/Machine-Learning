"""
Semi-Supervised Learning Demo
Techniques demonstrated:
1. Supervised baseline
2. Pseudo-labeling / Self-training
3. Simple two-view co-training demonstration

Run:
    python semi_supervised_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "iris_dataset.csv"

RANDOM_STATE = 42
LABELED_FRACTION = 0.20
CONFIDENCE_THRESHOLD = 0.90


def load_data():
    """Load the supplied CSV dataset."""
    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    X = df[feature_columns].values
    y = df["target"].values

    return X, y


def make_semi_supervised_split(X_train, y_train):
    """
    Hide most training labels to simulate a semi-supervised setting.
    The labels are retained internally only for evaluation/verification.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    labeled_count = max(
        1,
        int(len(X_train) * LABELED_FRACTION)
    )

    labeled_indices = rng.choice(
        len(X_train),
        size=labeled_count,
        replace=False
    )

    mask = np.zeros(len(X_train), dtype=bool)
    mask[labeled_indices] = True

    X_labeled = X_train[mask]
    y_labeled = y_train[mask]

    X_unlabeled = X_train[~mask]
    y_unlabeled_hidden = y_train[~mask]

    return (
        X_labeled,
        y_labeled,
        X_unlabeled,
        y_unlabeled_hidden
    )


def train_supervised_baseline(
    X_labeled_scaled,
    y_labeled,
    X_test_scaled,
    y_test
):
    """Train using only the small labeled set."""
    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(X_labeled_scaled, y_labeled)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def pseudo_labeling(
    X_labeled_scaled,
    y_labeled,
    X_unlabeled_scaled,
    X_test_scaled,
    y_test
):
    """
    Pseudo-label unlabeled samples whose maximum predicted probability
    exceeds CONFIDENCE_THRESHOLD, then retrain.
    """

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(X_labeled_scaled, y_labeled)

    probabilities = model.predict_proba(X_unlabeled_scaled)
    pseudo_labels = model.predict(X_unlabeled_scaled)

    confidence = np.max(probabilities, axis=1)

    confident_mask = confidence >= CONFIDENCE_THRESHOLD

    X_pseudo = X_unlabeled_scaled[confident_mask]
    y_pseudo = pseudo_labels[confident_mask]

    X_combined = np.vstack(
        [X_labeled_scaled, X_pseudo]
    )

    y_combined = np.concatenate(
        [y_labeled, y_pseudo]
    )

    final_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    final_model.fit(X_combined, y_combined)

    predictions = final_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    return (
        final_model,
        accuracy,
        len(X_pseudo),
        confidence,
        pseudo_labels,
        confident_mask
    )


def co_training_demo(
    X_labeled,
    y_labeled,
    X_unlabeled,
    X_test,
    y_test
):
    """
    Educational two-view co-training demonstration.

    View 1 = sepal measurements
    View 2 = petal measurements

    Model 1 and Model 2 exchange high-confidence pseudo-labels.
    """

    # Two complementary feature views
    view1_labeled = X_labeled[:, :2]
    view1_unlabeled = X_unlabeled[:, :2]
    view1_test = X_test[:, :2]

    view2_labeled = X_labeled[:, 2:]
    view2_unlabeled = X_unlabeled[:, 2:]
    view2_test = X_test[:, 2:]

    model1 = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model2 = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model1.fit(view1_labeled, y_labeled)
    model2.fit(view2_labeled, y_labeled)

    # Model 1 teaches Model 2
    prob1 = model1.predict_proba(view1_unlabeled)
    pred1 = model1.predict(view1_unlabeled)

    conf1 = np.max(prob1, axis=1)
    mask1 = conf1 >= CONFIDENCE_THRESHOLD

    if np.any(mask1):
        model2.fit(
            np.vstack([
                view2_labeled,
                view2_unlabeled[mask1]
            ]),
            np.concatenate([
                y_labeled,
                pred1[mask1]
            ])
        )

    # Model 2 teaches Model 1
    prob2 = model2.predict_proba(view2_unlabeled)
    pred2 = model2.predict(view2_unlabeled)

    conf2 = np.max(prob2, axis=1)
    mask2 = conf2 >= CONFIDENCE_THRESHOLD

    if np.any(mask2):
        model1.fit(
            np.vstack([
                view1_labeled,
                view1_unlabeled[mask2]
            ]),
            np.concatenate([
                y_labeled,
                pred2[mask2]
            ])
        )

    # Final prediction: average probabilities from both views
    test_prob1 = model1.predict_proba(view1_test)
    test_prob2 = model2.predict_proba(view2_test)

    average_probabilities = (
        test_prob1 + test_prob2
    ) / 2.0

    final_predictions = np.argmax(
        average_probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        final_predictions
    )

    return (
        accuracy,
        np.sum(mask1),
        np.sum(mask2),
        final_predictions
    )


def main():
    print("=" * 70)
    print("      SEMI-SUPERVISED LEARNING - VS CODE PRACTICAL")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    X, y = load_data()

    print(f"\nDataset: Iris")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print("Classes: Setosa, Versicolor, Virginica")

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------
    # Create labeled/unlabeled training data
    # --------------------------------------------------
    (
        X_labeled,
        y_labeled,
        X_unlabeled,
        hidden_labels
    ) = make_semi_supervised_split(
        X_train,
        y_train
    )

    print("\nSemi-supervised setup")
    print("-" * 30)
    print(f"Labeled training samples:   {len(X_labeled)}")
    print(f"Unlabeled training samples: {len(X_unlabeled)}")
    print(f"Test samples:               {len(X_test)}")
    print(f"Confidence threshold:       {CONFIDENCE_THRESHOLD}")

    # --------------------------------------------------
    # Scale features
    # --------------------------------------------------
    scaler = StandardScaler()

    X_labeled_scaled = scaler.fit_transform(
        X_labeled
    )

    X_unlabeled_scaled = scaler.transform(
        X_unlabeled
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # 1. Supervised baseline
    # --------------------------------------------------
    _, baseline_accuracy = train_supervised_baseline(
        X_labeled_scaled,
        y_labeled,
        X_test_scaled,
        y_test
    )

    print("\n1. SUPERVISED BASELINE")
    print("-" * 30)
    print(
        f"Accuracy using only labeled data: "
        f"{baseline_accuracy:.4f}"
    )

    # --------------------------------------------------
    # 2. Pseudo-labeling / self-training
    # --------------------------------------------------
    (
        final_model,
        ssl_accuracy,
        pseudo_count,
        confidence,
        pseudo_labels,
        confident_mask
    ) = pseudo_labeling(
        X_labeled_scaled,
        y_labeled,
        X_unlabeled_scaled,
        X_test_scaled,
        y_test
    )

    print("\n2. PSEUDO-LABELING / SELF-TRAINING")
    print("-" * 30)
    print(
        f"High-confidence pseudo-labels added: "
        f"{pseudo_count}"
    )
    print(
        f"Accuracy after pseudo-labeling: "
        f"{ssl_accuracy:.4f}"
    )

    # --------------------------------------------------
    # Show selected pseudo-labels
    # --------------------------------------------------
    if pseudo_count > 0:
        print("\nFirst 10 pseudo-labels:")
        print("Index | Predicted class | Confidence")
        print("-" * 42)

        class_names = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        selected_indices = np.where(
            confident_mask
        )[0][:10]

        for idx in selected_indices:
            print(
                f"{idx:5d} | "
                f"{class_names[pseudo_labels[idx]]:15s} | "
                f"{confidence[idx]:.4f}"
            )

    # --------------------------------------------------
    # 3. Co-training
    # --------------------------------------------------
    (
        cotrain_accuracy,
        model1_added,
        model2_added,
        cotrain_predictions
    ) = co_training_demo(
        X_labeled,
        y_labeled,
        X_unlabeled,
        X_test,
        y_test
    )

    print("\n3. CO-TRAINING")
    print("-" * 30)
    print(
        f"Model 1 confident samples shared: "
        f"{model1_added}"
    )
    print(
        f"Model 2 confident samples shared: "
        f"{model2_added}"
    )
    print(
        f"Co-training accuracy: "
        f"{cotrain_accuracy:.4f}"
    )

    # --------------------------------------------------
    # Detailed report for final pseudo-label model
    # --------------------------------------------------
    final_predictions = final_model.predict(
        X_test_scaled
    )

    print("\n4. CLASSIFICATION REPORT")
    print("-" * 30)

    print(
        classification_report(
            y_test,
            final_predictions,
            target_names=[
                "Setosa",
                "Versicolor",
                "Virginica"
            ]
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            final_predictions
        )
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)

    print(
        f"Supervised baseline       : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Pseudo-label / self-train : "
        f"{ssl_accuracy:.4f}"
    )

    print(
        f"Co-training               : "
        f"{cotrain_accuracy:.4f}"
    )

    print("\nNote:")
    print(
        "Semi-supervised learning does not guarantee "
        "higher accuracy in every dataset."
    )
    print(
        "The benefit depends on the quality of the "
        "unlabeled data, model, threshold and assumptions."
    )


if __name__ == "__main__":
    main()
