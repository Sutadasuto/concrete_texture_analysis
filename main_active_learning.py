### Set seeds for reproducibility
from numpy.random import seed

seed(0)
import tensorflow as tf

tf.random.set_seed(0)
###


import matplotlib.pyplot as plt
import numpy as np

from joblib import dump
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from shutil import copy
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

### My libraries
# Import user parameters from config file
import config

config_dict = config.__dict__

from config import *
from data import get_dataset, test_and_evaluate_model,train_test_xywnames_split
from recursive_feature_elimination_with_cross_validation import get_best_feats
from utils import create_model
###


# Process images to get dataset
print("Building dataset from path... ")
config_dict[
    "only_consensus_images"] = False  # Force the code to include all the images even if the config file states otherwise
X, y, w, names, feature_names = get_dataset(**config_dict)

# Find the images with and without consensus
consensus_indices = np.where(w.ravel() == 1.0)[0]
non_consensus_indices = np.where(w.ravel() < 1.0)[0]

# Split data
print("Building train and test split from consensus images... ")
X_train, X_test, y_train, y_test, weights_train, weights_test, names_train, names_test = train_test_xywnames_split(
    X[consensus_indices, :], y[consensus_indices, :], w[consensus_indices, :], names[consensus_indices, :])
X_pool = X[non_consensus_indices, :]
y_pool = y[non_consensus_indices, :]
weights_pool = w[non_consensus_indices, :]
names_pool = names[non_consensus_indices, :]
print("Done.\n")

# Show class frequencies in the initial training split
plt.figure('Frequencies of classes in training split')
label_names, label_counts = np.unique(y_train, return_counts=True)
plt.bar(label_names, label_counts)
plt.show(block=False)

# Show n representative images per class from the images with consensus
n_representative_images_per_class = 4
# image_paths = np.array((len(label_names), n_representative_images_per_class), np.str)
fig, m_axs = plt.subplots(len(label_names), n_representative_images_per_class,
                          figsize=(4 * n_representative_images_per_class, 3 * 7), num='Class Examples')
for i, label_name in enumerate(label_names):
    m_axs[i, 0].set_title(label_name)
    candidate_indices = np.where((y_train == label_name) & (weights_train == 1.0))[0]
    np.random.shuffle(candidate_indices)
    for j in range(n_representative_images_per_class):
        try:
            image_name = names_train[candidate_indices[j], 0]
        except IndexError:
            print("Warning: %s images cannot be extracted from class '%s' because the number of samples is %s" % (j+1, label_name, len(candidate_indices)))
            break
        image = plt.imread(os.path.join(photo_dir, image_name))
        m_axs[i, j].imshow(image, cmap='gray')
        m_axs[i, j].axis("off")
    plt.show(block=False)

# Scale features
if scale_features:
    print("Fitting standard scaler on train split... ")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_pool = scaler.transform(X_pool)
    X = scaler.transform(X)
    dump(scaler, os.path.join(results_path, "data_scaler.joblib"))
    print("Done.\n")

# Get class weights for class imbalance. Ignoring consensus weight because the initial samples were consensual
if class_weight:  # Either None or 'balanced', as defined in the config file
    class_weights = compute_class_weight(class_weight=class_weight, classes=label_names, y=y_train.ravel())
else:
    class_weights = np.ones(len(label_names))
# Modify sample weight according to its class' weight
weights_train = np.array([class_weights[np.where(label_names == label)] for label in y_train.ravel()])
weights_test = np.array([class_weights[np.where(label_names == label)] for label in y_test.ravel()])
weights_pool = np.array([class_weights[np.where(label_names == label)] for label in y_pool.ravel()])

# Perform recursive feature elimination over the training split
if only_best_feats:
    plt.figure("RFE")
    print("Performing recursive feature elimination... ")
    best_feats = get_best_feats(X_train, np.ravel(y_train), feat_elimination_clf, folds, min_features_to_select, plot=True,
                                step=feature_step, sample_weights=weights_train)
    dump(best_feats, os.path.join(results_path, "best_features.joblib"))
    plt.savefig(os.path.join(results_path, "clf_rfe.png"))
    plt.show(block=False)
else:
    best_feats = np.array([True] * X.shape[-1])
X_train = X_train[:, best_feats]
X_test = X_test[:, best_feats]
X_pool = X_pool[:, best_feats]

### Create baseline model
## Encode the class names to a [n_samples, n_classes] one-hot vector
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
dump(enc, os.path.join(results_path, "class_encoder.joblib"))
y_train, y_test, y_pool = [enc.transform(y_train).toarray(), enc.transform(y_test).toarray(), enc.transform(y_pool).toarray()]
## Create callbacks
# Early stop on validation loss plateau
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, verbose=0)
# Save the model with the minimum validation lost
weights_file_name = "weights.hdf5"
checkpoint = ModelCheckpoint(os.path.join(results_path, weights_file_name), monitor='val_loss', verbose=0, \
                             save_best_only=True, save_weights_only=True, \
                             mode='min', save_frequency=1)
clf = KerasClassifier(build_fn=create_model, input_size=X_train.shape[-1], output_size=len(label_names),
                      **training_parameters)
model = create_model(input_size=X_train.shape[-1], output_size=len(label_names))
fit_parameters = {'sample_weight': weights_train.ravel(), 'callbacks': [checkpoint, early_stop], 'validation_data': (X_test, y_test),
                  'verbose': 0}

# Initialize the active learner
learner = ActiveLearner(
    estimator=clf,
    query_strategy=uncertainty_sampling,
    X_train=X_train, y_train=y_train,
)
learner.fit(X_train, y_train, **fit_parameters)

n_queries = 60


def get_score(learner, class_encoder, X, y, w=None):
    y_pred = learner.predict(X)
    if not w is None:
        score = f1_score(class_encoder.inverse_transform(y), class_encoder.inverse_transform(y_pred), average='macro',
                        sample_weight=w.ravel())
    else:
        score = f1_score(class_encoder.inverse_transform(y), class_encoder.inverse_transform(y_pred), average='macro')
    return score

model.load_weights(os.path.join(results_path, weights_file_name))
scores = [get_score(model, enc, X_test, y_test, weights_test)]
dictOfWords = {i: label_names[i] for i in range(len(label_names))}

X_learn, y_learn = [], []
cnt_conseq_correct = 0
new_labels = []
for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    query_img_file = names_pool[query_idx[0], 0]
    plt.figure('Query', figsize=(10, 5));
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(os.path.join(photo_dir, query_img_file)), cmap='gray')
    plt.title('Texture to label %s' % query_img_file)
    plt.subplot(1, 2, 2)
    plt.title('F-1 macro of your model')
    plt.plot(range(i + 1), scores)
    plt.scatter(range(i + 1), scores)
    plt.xlabel('number of queries')
    plt.ylabel('accuracy')
    plt.xlim([0, n_queries])
    plt.pause(0.05)
    plt.savefig(os.path.join(results_path, 'active_score_progression.png'))

    plt.figure('Mosaic view');
    plt.clf()
    mosaic_img_file = query_img_file[0:query_img_file.find('-')] + '.tiff'
    plt.imshow(plt.imread(os.path.join(photo_dir, mosaic_img_file)), cmap='gray')
    plt.title('Window %i needs labeling' % int(query_img_file[query_img_file.find('-') + 1:query_img_file.find('.')]))
    plt.pause(0.05)

    print("What class is this texture %s ?" % dictOfWords)
    y_learner = enc.inverse_transform(learner.predict_proba(query_inst.reshape(1, -1), **{'verbose': 0}))[0][0]
    print("Learner predicts '%s'" % y_learner)

    y_new = len(dictOfWords.keys())  # check out the number of classes we have
    while not y_new in dictOfWords.keys():  # repeat until we have a valid class
        y_new = input()
        if y_new == '':  # on Enter oracle accepts the model's prediction
            y_new = list(dictOfWords.keys())[list(dictOfWords.values()).index(y_learner)]
            print(y_new)
        else:
            y_new = int(y_new)
            if (y_new) < 0:
                break

    if int(y_new) < 0:  # quit active learning loop
        print('OK. Leaving the learning loop on user\'s demand!')
        break
    y_new = dictOfWords[int(y_new)]
    if y_new == y_learner:
        cnt_conseq_correct += 1
        print("Conseq. correct %i" % cnt_conseq_correct)
        if cnt_conseq_correct >= 10:
            break
    else:
        cnt_conseq_correct = 0

    # Get new class weight according to the expanded train set
    current_y_train = np.concatenate((enc.inverse_transform(learner.y_training), [[y_new]]), axis=0)
    if class_weight:
        class_weights = compute_class_weight(class_weight=class_weight, classes=label_names,
                                             y=current_y_train.ravel()
                                             )
    # Modify sample weight according to its class' weight. Ignoring consensus weight because the initial samples were consensual
    weights_train = np.array([class_weights[np.where(label_names == label)] for label in current_y_train.ravel()])
    weights_test = np.array([class_weights[np.where(label_names == enc.inverse_transform(label[None, ...])[0][0])] for label in y_test])
    weights_pool = np.array([class_weights[np.where(label_names == enc.inverse_transform(label[None, ...])[0][0])] for label in y_pool])

    # Fine-tune the learner with the new sample AND the new class weights
    only_new = False
    if only_new:
        fit_parameters['sample_weight'] = weights_train[-1, 0].ravel()
    else:
        fit_parameters['sample_weight'] = weights_train.ravel()
    # Re-initilize callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, verbose=0)
    cp = ModelCheckpoint(os.path.join(results_path, weights_file_name), monitor='val_loss', verbose=0, \
                                 save_best_only=True, save_weights_only=True, \
                                 mode='min', save_frequency=1)
    fit_parameters['callbacks'] = [es, cp]
    learner.teach(query_inst, enc.transform([[y_new]]).toarray(), only_new=only_new, **fit_parameters)  # To discuss

    new_labels.append([names_pool[query_idx[0], 0], y_new])
    X_pool = np.delete(X_pool, query_idx[0], axis=0)
    y_pool = np.delete(y_pool, query_idx[0], axis=0)
    names_pool = np.delete(names_pool, query_idx[0], axis=0)
    weights_pool = np.delete(weights_pool, query_idx[0])
    model.load_weights(os.path.join(results_path, weights_file_name))
    scores.append(get_score(model, enc, X_test, y_test, weights_test))

print('********************************************************************')
print('* %d consequtive predictions achieved. Leaving the learning loop! *' % cnt_conseq_correct)
print('********************************************************************')
# pdb.set_trace()

plt.figure("Validation F-score history")
plt.title('F-1 macro of your model')
plt.plot(range(n_queries + 1), scores)
plt.scatter(range(n_queries + 1), scores)
plt.xlabel('number of queries')
plt.ylabel('score')
plt.xlim([0, n_queries])
plt.savefig(os.path.join(results_path, "f_history.png"))
plt.show(block=False)

# Evaluate the trained model, plot and save confusion matrix as image
model.load_weights(os.path.join(results_path, weights_file_name))
test_and_evaluate_model(model, enc, X_test, y_test, names_test, photo_dir, weights_test, results_path)

with open(os.path.join(results_path, "active_labels.csv"), "w+") as f:
    f.write("\n".join([",".join(pair) for pair in new_labels]))

# Save the config file to preserve user parameters
copy("config.py", os.path.join(results_path, "config.py"))

