### Set seeds for reproducibility
import matplotlib.pyplot as plt
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)
###

### My libraries
from data import calculate_features
from recursive_feature_elimination_with_cross_validation import *
from see_classification import save_keras_classification_comparison
###

### Data processing
import re
from shutil import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Embedding techniques comparison
from manifold_learning import embedding_techniques_comparison
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
###

### Data training
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import f1_score as metric
###

# Paths of the dataset and the csv file containing the labels
photo_dir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows_test"
label_path = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows/malo_rodrigo_eva.csv"
only_consensus_images = True

# Save results to this location
results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Define search parameters
training_split = 0.8  # Amount of data to use for training

# Recursive feature elimination with cross-validation parameters
feat_elimination_clf = LogisticRegression(random_state=0, max_iter=1000)  # Baseline model for recursive feature elimination
min_features_to_select = 10  # Minimum number of features to consider
folds = 5  # Folds used for cross-validation feature elimination
feature_step = 2  # Features to eliminate per iteration

# Define embeddings to test. For more examples of embeddings, see: https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#embedding-techniques-comparison
n_neighbors = 30
embeddings = {
    "MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
    "t-SNE embeedding": TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    ),
}

# GLCM parameters (number_of_features = len(distances) * len(angles) * len(props))
distances = list(range(1, 42, 10))  # Offset distance
angles = [0, pi/2]  # Offset direction (in radians)
standardize_glcm_image = True  # Standardize the image so that the GLCM levels correspond to the range [μ-3.1σ, μ+3.1σ] in the input image
glcm_levels = 11  # Number of intensity bins to calculate the GLCM (the resulting matrix size is glcm_levels×glcm_levels)
props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']  # Properties to calculate from a GLCM according to scikit-image documentation

# LBP parameters (number_of_features = len(ps) * len(radii) * bins)
ps = [8]  # Number of neighbors to calculate the binary pattern. Transformed pixels have a value [0, 2^neighbors]
radii = list(range(1, 21, 5))  # Offset radius from the pixel to its neighbors
standardize_lbp_image = True  # The input image is standardized from [μ-3.1σ, μ+3.1σ] to [0, lbp_levels-1] (see line below)
lbp_levels = 11  # Number of intensity bins in the image used to calculate the LBP
bins = 16  # The histogram of intensities from the transformed image is calculated using this number of bins

# Process images to get dataset
print("Building dataset from path... ")
expert = "Malo"  # Annotations of single person to consider if agreement_weights is 'None' or if there is not a winner class in the vector of labels for a single image
agreement_weights = [1.0, 0.5, 0.25]  # Weights for consensus, majority, and expert tie-breaking
ignore_nd = True
X, y, w, names = calculate_features([photo_dir, label_path],
                                 distances=distances, angles=angles, standardize_glcm_image=standardize_glcm_image, glcm_levels=glcm_levels, props=props,
                                 ps=ps, radii=radii, standardize_lbp_image=standardize_lbp_image, lbp_levels=lbp_levels, bins=bins,
                                 ignore_nd=ignore_nd, expert=expert, agreement_weights=agreement_weights)  # Calculate using skimage
# The segment of code below can be used to write the estimated labels and weights in a csv file
def key(value):
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))
sorted_names = sorted(names.ravel(), key=key)
sorted_indices = [sorted_names.index(name) for name in names.ravel()]
sorted_array = np.concatenate([names, y, w], axis=-1)[np.argsort(sorted_indices), :]
with open(os.path.join(results_path, "annotation_agreement_weights.csv"), "w+") as f:  # Write to csv
    string = "image,label,weight\n"
    string += "\n".join([",".join(row) for row in sorted_array.tolist()])
    f.write(string)
print("Done.\n")

if only_consensus_images:
    consensus_images_indices = np.where(w.ravel() == 1.0)[0]
    X = X[consensus_images_indices, :]
    y = y[consensus_images_indices, :]
    w = w[consensus_images_indices, :]
    names = names[consensus_images_indices, :]

# Split data
print("Building train and test split... ")
X_train, X_test, y_train, y_test = train_test_split(np.array([[i] for i in range(X.shape[0])]), y, train_size=training_split, stratify=y, random_state=0)
names_train = names[np.ravel(X_train), :]
names_test = names[np.ravel(X_test), :]
weights_train = w[np.ravel(X_train), :]
weights_test = w[np.ravel(X_test), :]
X_train = X[np.ravel(X_train), :]
X_test = X[np.ravel(X_test), :]
print("Done.\n")

# Scale features
print("Fitting standard scaler on train split... ")
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("Done.\n")

# Perform recursive feature elimination over the training split
print("Performing recursive feature elimination... ")
best_feats = get_best_feats(X_train, np.ravel(y_train), feat_elimination_clf, folds, min_features_to_select, plot=True, step=feature_step)
plt.savefig(os.path.join(results_path, "clf_rfe.png"))
plt.show()

### Embedding techniques comparison
embedding_techniques_comparison(X, y, names, embeddings, photo_dir)
plt.show()

# Define the neural network
def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[F1Score(num_classes=output_size, average="macro")])
    return model

# Encode the class names to a [n_samples, n_classes] one-hot vector
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)
y_train, y_test = [enc.transform(y_train).toarray(), enc.transform(y_test).toarray()]
n_class = y_train.shape[1]

n_feat = X_train[:, best_feats].shape[1]  # Number of features after recursive feature elimination

# Create the model with the scikit-learn wrapper
# A regressor is used to return a vector of probabilities ratter than a class
clf = KerasRegressor(build_fn=create_model,
                     input_size=n_feat, output_size=n_class,
                     epochs=2000,
                     batch_size=32,
                     verbose=1)

# Train the model, using a subset of the training split for validation
clf.fit(X_train[:, best_feats], y_train, validation_split=0.10, sample_weight=weights_train.ravel())

# Evaluate the trained model
y_pred = clf.predict(X_test[:, best_feats])  # Predict the test split
disp = ConfusionMatrixDisplay.from_predictions(enc.inverse_transform(y_test), enc.inverse_transform(y_pred), normalize='true')  # Calculate confusion matrix
score = metric(enc.inverse_transform(y_test), enc.inverse_transform(y_pred), average='macro')  # Calculate the macro F-1
# Plot confusion matrix
plt.title("Best classifier: %s (%s)" % (str(type(clf)).split("'")[1].split(".")[-1], round(score, 4)))
disp.im_.set_clim(0, 1)
plt.savefig(os.path.join(results_path, "cm.png"))
plt.show()
plt.clf()

# Save confusion matrix as image
print("\nSaving predictions to '%s'" % results_path)
save_keras_classification_comparison(clf, X_test[:, best_feats], y_test, names_test, photo_dir, results_path, enc)
# Save the current script (lazy way of saving the parameters used)
copy("main.py", os.path.join(results_path, "code.py"))
print("Done.\n")
