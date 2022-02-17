import csv
import cv2
import numpy as np
import os

from math import pi
from shutil import copy
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score as metric
from sklearn.model_selection import train_test_split

from lbp_tools import *
from glcm_tools import *
from see_classification import save_keras_classification_comparison
from utils import key

# def get_dataset(mode, paths, **kwargs):
#     if mode == "folder":
#
#         # photo_dir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows"
#         # label_path = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows/labels.csv"
#         photo_dir = paths[0]
#         label_path = paths[1]
#
#         # distances = list(range(1, 21))
#         # angles = [0, pi / 2]
#         # levels = 8
#         # props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
#         distances = kwargs["distances"]
#         angles = kwargs["angles"]
#         levels = kwargs["levels"]
#         props = kwargs["props"]
#
#         img_names = sorted([f for f in os.listdir(photo_dir)
#                             if not f.startswith(".") and os.path.isfile(os.path.join(photo_dir, f))],
#                            key=lambda f: f.lower())
#
#         with open(label_path, "r") as f:
#             labels = np.array([line.split(",") for line in f.readlines()])
#
#         labels = labels[1:, :2]
#
#         names = []
#         X = []
#         Y = []
#         for name in img_names:
#             idx = np.where(labels[:, 0] == name)
#             if idx[0].size == 0:
#                 continue
#             else:
#                 localLabels = labels[idx, 1][0, 0].split("+")
#                 for label in localLabels:
#                     if label.strip() == "":
#                         continue
#                     Y.append([label.strip()])
#                     names.append([name])
#                     img = cv2.imread(os.path.join(photo_dir, name), cv2.IMREAD_GRAYSCALE)
#                     glcms, bin_image = region_glcm(img, distances, angles, levels)
#                     features = get_glcm_features(glcms, props, avg_and_range=True)
#                     X.append(features[0, :])
#
#         names = np.array(names)
#         X = np.array(X)
#         Y = np.array(Y)
#
#     elif mode == "csv":
#
#         X = np.genfromtxt(paths[0], delimiter=',')
#         with open(paths[1], "r") as f:
#             Y = np.array([i for i in csv.reader(f)])
#         with open(paths[2], "r") as f:
#             names = np.array([i for i in csv.reader(f)])
#
#     else:
#         raise ValueError("Invalid mode.")
#
#     return X, Y, names


def get_dataset(**kwargs):
    X, y, w, names, feature_names = calculate_features([kwargs["photo_dir"], kwargs["label_path"]],
                                        distances=kwargs["distances"], angles=kwargs["angles"],
                                        standardize_glcm_image=kwargs["standardize_glcm_image"], glcm_levels=kwargs["glcm_levels"],
                                        props=kwargs["props"],
                                        ps=kwargs["ps"], radii=kwargs["radii"], standardize_lbp_image=kwargs["standardize_lbp_image"],
                                        lbp_levels=kwargs["lbp_levels"], bins=kwargs["bins"],
                                        ignore_nd=kwargs["ignore_nd"], expert=kwargs["expert"],
                                        agreement_weights=kwargs["agreement_weights"])  # Calculate using skimage
    # The segment of code below can be used to write the estimated labels and weights in a csv file
    save_agreement_labels = True
    if save_agreement_labels:
        sorted_names = sorted(names.ravel(), key=key)
        sorted_indices = [sorted_names.index(name) for name in names.ravel()]
        sorted_array = np.concatenate([names, y, w], axis=-1)[np.argsort(sorted_indices), :]
        with open(os.path.join(kwargs["results_path"], "annotation_agreement_weights.csv"), "w+") as f:  # Write to csv
            string = "image,label,weight\n"
            string += "\n".join([",".join(row) for row in sorted_array.tolist()])
            f.write(string)
        print("Done.\n")

    if kwargs["only_consensus_images"]:
        consensus_images_indices = np.where(w.ravel() == 1.0)[0]
        X = X[consensus_images_indices, :]
        y = y[consensus_images_indices, :]
        w = w[consensus_images_indices, :]
        names = names[consensus_images_indices, :]

    return X, y, w, names, feature_names


def label_convention(label, labels_to_ignore):
    if label == "air-bubble":  # Decided to ignore air bubbles as they are more of localized defects rather thann texture defects
        label = "nd"
    if label == "nd":
        label = "decision-null"
    # if label == "micro-cracking":
    #     label = "shark-skin"
    if label.strip() in labels_to_ignore:
        return None
    return label


def calculate_features(paths, **kwargs):
    photo_dir = paths[0]
    label_path = paths[1]

    img_names = sorted([f for f in os.listdir(photo_dir)
                        if not f.startswith(".") and os.path.isfile(os.path.join(photo_dir, f))],
                       key=lambda f: f.lower())

    with open(label_path, "r") as f:
        labels = np.array([line.strip().split(",") for line in f.readlines()])
        props = kwargs["props"]

    # Multi-annotator setup
    try:
        expert = kwargs["expert"]
    except KeyError:
        expert = labels[0, 1]
    try:
        agreement_weights = kwargs["agreement_weights"]
    except KeyError:
        agreement_weights = None
    expert_column = np.where(labels[0, :] == expert)[0][0]
    labels = labels[1:, [0, expert_column]] if not agreement_weights else labels[1:, :]
    if not agreement_weights:
        agreement_weights = [1.0]  # All the samples will have the same weight
    n_annotators = labels.shape[-1] - 1

    labels_to_ignore = [""]
    try:
        ignore_nd = kwargs["ignore_nd"]  # Flag to ignore 'nd' class
    except KeyError:
        ignore_nd = False

    # GLCM parameters
    distances = kwargs["distances"]
    angles = kwargs["angles"]
    standardize_glcm_image = kwargs["standardize_glcm_image"]
    glcm_levels = kwargs["glcm_levels"]

    # LBP parameters
    ps = kwargs["ps"]
    radii = kwargs["radii"]
    standardize_lbp_image = kwargs["standardize_lbp_image"]
    lbp_levels = kwargs["lbp_levels"]
    bins = kwargs["bins"]

    names = []
    X = []
    Y = []
    W = []
    for name in img_names:
        idx = np.where(labels[:, 0] == name)
        if idx[0].size == 0:
            continue
        else:
            n_annotators = 0
            vote_dict = {}
            image_labels = labels[idx[0], 1:][0]
            for annotator_labels in image_labels:
                annotator_label = annotator_labels.split("+")
                for label in annotator_label:
                    label = label_convention(label.strip(), labels_to_ignore)
                    if not label:
                        continue
                    if not label in vote_dict.keys():
                        vote_dict[label] = 1
                    else:
                        vote_dict[label] += 1
                    n_annotators += 1
            try:
                max_votes = max(vote_dict.values())
            except ValueError:
                continue
            most_voted_label = [label for label in vote_dict.keys() if vote_dict[label] == max_votes]

            for label in most_voted_label:
                complete_disagreement = False
                if max_votes == n_annotators:
                    W.append([agreement_weights[0]])
                    Y.append([label])
                elif len(most_voted_label) < n_annotators:
                    W.append([agreement_weights[1]])
                    Y.append([label])
                else:
                    complete_disagreement = True
                    expert_label = label_convention(
                        labels[idx[0], expert_column][0].split("+")[0],  # The first label in the expert column
                        labels_to_ignore
                    )
                    W.append([agreement_weights[2]])
                    Y.append([expert_label])
                if Y[-1] == ["decision-null"] and ignore_nd:
                    W.pop(-1)
                    Y.pop(-1)
                    continue

                names.append([name])
                img = cv2.imread(os.path.join(photo_dir, name), cv2.IMREAD_GRAYSCALE)
                img = delete_black_regions(img)
                glcms, bin_image = region_glcm(img, distances, angles, glcm_levels, standardize=standardize_glcm_image)
                glcm_features = get_glcm_features(glcms, props)
                lbps = region_lbp(img, radii, ps, lbp_levels, standardize=standardize_lbp_image)
                lbps_features = get_lbp_histograms(lbps, bins)
                X.append(np.concatenate((glcm_features[0, :], lbps_features[0, :])))

                if complete_disagreement:
                    break

    glcm_feature_names = get_glcm_feature_names(distances, angles, props)
    lbps_feature_names = get_lbp_feature_names(radii, ps, bins)
    feature_names = glcm_feature_names + lbps_feature_names
    names = np.array(names)
    X = np.array(X)
    Y = np.array(Y)
    W = np.array(W)

    return X, Y, W, names, np.array(feature_names)[None, ...]


def delete_black_regions(img):
    binary_image = img > 0
    h, w = binary_image.shape

    for row in range(h):
        if np.sum(binary_image[row, :]) == w:
            first_row = row
            break
    for row in reversed(range(h)):
        if np.sum(binary_image[row, :]) == w:
            last_row = row + 1
            break

    binary_image = binary_image[first_row:last_row, :]
    h, w = binary_image.shape

    for col in range(w):
        if np.sum(binary_image[:, col]) == h:
            first_col = col
            break
    for col in reversed(range(w)):
        if np.sum(binary_image[:, col]) == h:
            last_col = col + 1
            break

    return img[first_row:last_row, first_col:last_col]


def test_and_evaluate_model(model, class_encoder, x, y, names, photo_dir, sample_weights=None, results_path="results"):
    if sample_weights is None:
        sample_weights = [1.0 for i in range(len(y))]
    # Evaluate the trained model
    y_pred = model.predict(x)  # Predict the test split
    if len(y_pred.shape) == 1:  # i.e. predicting a single class rather than all class probabilities
        y_pred = model.predict_proba(x)
    disp = ConfusionMatrixDisplay.from_predictions(class_encoder.inverse_transform(y), class_encoder.inverse_transform(y_pred),
                                                   normalize='true',
                                                   sample_weight=sample_weights.ravel())  # Calculate confusion matrix
    score = metric(class_encoder.inverse_transform(y), class_encoder.inverse_transform(y_pred), average='macro',
                   sample_weight=sample_weights.ravel())  # Calculate the macro F-1
    # Plot confusion matrix
    plt.title("Best classifier: %s (%s)" % (str(type(model)).split("'")[1].split(".")[-1], round(score, 4)))
    disp.im_.set_clim(0, 1)
    plt.savefig(os.path.join(results_path, "cm.png"))
    plt.show()

    # Save confusion matrix as image
    print("\nSaving predictions to '%s'" % results_path)
    save_keras_classification_comparison(model, x, y, sample_weights, names, photo_dir, results_path, class_encoder)
    print("Done.\n")


def train_test_xywnames_split(X, y, w, names, training_split=0.8):
    X_train, X_test, y_train, y_test = train_test_split(np.array([[i] for i in range(X.shape[0])]), y,
                                                        train_size=training_split, stratify=y, random_state=0)
    names_train = names[np.ravel(X_train), :]
    names_test = names[np.ravel(X_test), :]
    weights_train = w[np.ravel(X_train), :]
    weights_test = w[np.ravel(X_test), :]
    X_train = X[np.ravel(X_train), :]
    X_test = X[np.ravel(X_test), :]

    return X_train, X_test, y_train, y_test, weights_train, weights_test, names_train, names_test