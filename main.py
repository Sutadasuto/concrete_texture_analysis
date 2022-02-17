### Set seeds for reproducibility
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)
###

### My libraries
from data import test_and_evaluate_model, get_dataset, train_test_xywnames_split
from recursive_feature_elimination_with_cross_validation import *
###

### Data processing
import numpy as np
from joblib import dump
from shutil import copy
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
# Embedding techniques comparison
from manifold_learning import embedding_techniques_comparison
###

### Data training
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import create_model
###

# Import user parameters from config file
import config
config_dict = config.__dict__
from config import *

# Process images to get dataset
print("Building dataset from path... ")
X, y, w, names, feature_names = get_dataset(**config_dict)

# Split data
print("Building train and test split... ")
X_train, X_test, y_train, y_test, weights_train, weights_test, names_train, names_test = train_test_xywnames_split(X, y, w, names)
print("Done.\n")

# Scale features
if scale_features:
    print("Fitting standard scaler on train split... ")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X = scaler.transform(X)
    dump(scaler, os.path.join(results_path, "data_scaler.joblib"))
    print("Done.\n")

# Get class weights for class imbalance
class_names = np.unique(y_train)
if class_weight:
    class_weights = compute_class_weight(class_weight=class_weight, classes=class_names, y=y_train.ravel())
else:
    class_weights = np.ones(len(class_names))
# Modify sample weigh according to its class' weight
train_sample_class_weights = [class_weights[np.where(class_names == label)] for label in y_train.ravel()]
weights_train = np.array(train_sample_class_weights) * weights_train
test_sample_class_weights = [class_weights[np.where(class_names == label)] for label in y_test.ravel()]
weights_test = np.array(test_sample_class_weights) * weights_test


# Perform recursive feature elimination over the training split
if only_best_feats:
    print("Performing recursive feature elimination... ")
    best_feats = get_best_feats(X_train, np.ravel(y_train), feat_elimination_clf, folds, min_features_to_select, plot=True,
                                step=feature_step, sample_weights=weights_train)
    dump(best_feats, os.path.join(results_path, "best_features.joblib"))
    plt.savefig(os.path.join(results_path, "clf_rfe.png"))
    plt.show()
else:
    best_feats = np.array([True] * X.shape[-1])

### Embedding techniques comparison
embedding_techniques_comparison(X, y, names, embeddings, photo_dir, fig_size=(45, 22.5))
plt.savefig(os.path.join(results_path, "2d_projection.png"))
plt.show()

# Encode the class names to a [n_samples, n_classes] one-hot vector
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y_train, y_test = [enc.transform(y_train).toarray(), enc.transform(y_test).toarray()]
n_class = y_train.shape[1]

n_feat = X_train[:, best_feats].shape[1]  # Number of features after recursive feature elimination

# Create the model
clf = create_model(input_size=n_feat, output_size=n_class)

# Early stop on validation loss plateau
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
# Save the model with the minimum validation lost
weights_file_name = "weights.hdf5"
checkpoint = ModelCheckpoint(os.path.join(results_path, weights_file_name), monitor='val_loss', verbose=1, \
                             save_best_only=True, save_weights_only=True, \
                             mode='min', save_frequency=1)
# Train the model, using a subset of the training split for validation
clf.fit(X_train[:, best_feats], y_train, validation_split=0.10,
        sample_weight=weights_train.ravel(),
        callbacks=[checkpoint, early_stop], **training_parameters)

# Evaluate the trained model, plot and save confusion matrix as image
test_and_evaluate_model(clf, enc, X_test[:, best_feats], y_test, names_test, photo_dir, weights_test, results_path)

# Save the config file to preserve user parameters
copy("config.py", os.path.join(results_path, "config.py"))