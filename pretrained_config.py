import os
from math import pi

from sklearn.manifold import MDS, TSNE
from sklearn.linear_model import LogisticRegression

# Paths of the dataset and the csv file containing the labels
photo_dir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows"
label_path = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows/malo_rodrigo_eva.csv"

# Parameters to generate dataset
expert = "Malo"  # Choose only the labels provided by this person if agreement_weights is 'None' or if there is not a majority label in the vector of labels for a single image
agreement_weights = [1.0, 0.5, 0.25]  # Weights for consensus, majority, and expert tie-breaking in 'label_path'
ignore_nd = True  # If True, samples with final label 'nd' will be ignored in the generated dataset
only_consensus_images = True  # If True, the generated dataset will contain only samples with a consensual label between annotators
class_weight = None  # None or 'balanced'. Weight each sample to give more importance to minority classes
scale_features = True   # If True, the training features will be standardized according to the training split
only_best_feats = True  # If True, recursive feature elimination will be done before training

# Save results to this location
results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Define search parameters
training_split = 0.8  # Amount of data to use for training

# Recursive feature elimination with cross-validation parameters
feat_elimination_clf = LogisticRegression(random_state=0, max_iter=1000, class_weight=class_weight)  # Baseline model for recursive feature elimination
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
radii = list(range(1, 17, 5))  # Offset radius from the pixel to its neighbors
standardize_lbp_image = True  # The input image is standardized from [μ-3.1σ, μ+3.1σ] to [0, lbp_levels-1] (see line below)
lbp_levels = 11  # Number of intensity bins in the image used to calculate the LBP
bins = 16  # The histogram of intensities from the transformed image is calculated using this number of bins

# NN training parameters
training_parameters = {'epochs': 2000, 'batch_size': 32, 'verbose': 1}