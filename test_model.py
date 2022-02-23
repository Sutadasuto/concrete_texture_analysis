model_results_path = "results_al"

import os
from shutil import copy
copy(os.path.join(model_results_path, "config.py"), "pretrained_config.py")
# Import user parameters from config file
import pretrained_config
from pretrained_config import *
config_dict = pretrained_config.__dict__

from joblib import load
from sklearn.preprocessing import OneHotEncoder

from data import get_dataset, test_and_evaluate_model
from utils import create_model

# Replace variable values to match the new data on which the model will be tested
# Paths of the dataset and the csv file containing the labels
photo_dir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows"
label_path = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows/malo_rodrigo_eva-no_consensus.csv"

# Parameters to generate dataset
config_dict["photo_dir"] = photo_dir
config_dict["label_path"] = label_path
config_dict["ignore_nd"] = True  # If True, samples with final label 'nd' will be ignored in the generated dataset
config_dict["only_consensus_images"] = False  # If True, the generated dataset will contain only samples with a consensual label between annotators

# Save results to this location
results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Process images to get dataset
print("Building dataset from path... ")
X, y, w, names, feature_names = get_dataset(**config_dict)

if os.path.exists(os.path.join(model_results_path, "data_scaler.joblib")):
    scaler = load(os.path.join(model_results_path, "data_scaler.joblib"))
    X = scaler.transform(X)

if os.path.exists(os.path.join(model_results_path, "best_features.joblib")):
    best_feats = load(os.path.join(model_results_path, "best_features.joblib"))
    X = X[:, best_feats]

# Encode the class names to a [n_samples, n_classes] one-hot vector
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y = enc.transform(y).toarray()

n_feat = X.shape[1]
n_class = y.shape[1]

# Create model and load pre-trained weights
clf = create_model(input_size=n_feat, output_size=n_class)
clf.load_weights(os.path.join(model_results_path, "weights.hdf5"))

# Evaluate the pre-trained model, plot and save confusion matrix as image
test_and_evaluate_model(clf, enc, X, y, names, photo_dir, w, results_path)

readme_string = "Config from: %s\nTested on images from: %s\n with labels from: %s" % (model_results_path, photo_dir, label_path)
with open(os.path.join(results_path, "_README"), "w+") as f:
    f.write(readme_string)
