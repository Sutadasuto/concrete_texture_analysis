import numpy as np

from statsmodels.stats.inter_rater import fleiss_kappa


def get_fleiss_kappa_from_csv(path_to_csv):

    with open(path_to_csv, "r") as f:
        labels = np.array([line.strip().split(",") for line in f.readlines()])

    # Convention on decisions taken
    labels[np.where(labels == "")] = "nd"  # Lack of annotation is considered "no decision"
    labels[np.where(labels == "air-bubble")] = "nd"  # Temporary ignore air bubbles since they are not texture issues
    labels[np.where(labels == "nd")] = "decision-null"  # For demonstration purposes, use this label to be the first alphabetically

    labels = labels[1:, 1:]  # Remove header and image names

    # labels = labels[:, [0, 1, 4]]  # Test with Malo, Rodrigo and Eva's annotations

    class_names = np.unique(labels)
    class_names = [name for name in class_names if not "+" in  name]  # Ignore composite classes

    table = np.zeros((labels.shape[0], len(class_names)))  # Create table for the kappa calculation

    # Fill the table
    for i, row in enumerate(labels):  # Check the "subjects" (images) one by one
        for j in range(len(row)):
            decision = row[j].split("+")  # Convert composite classes to individual ones by taking just the first class
            table[i, class_names.index(decision[0])] += 1

    agreement_score = fleiss_kappa(table)
    return agreement_score

# get_fleiss_kappa_from_csv("/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows/labels.csv")

