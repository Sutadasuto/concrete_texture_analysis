import cv2
import numpy as np
import os

from matplotlib import offsetbox
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def embedding_techniques_comparison(X, y, names, embeddings, photo_dir=None, fig_size=(30, 15)):

    # Helper function to plot embedding
    def plot_embedding(X, title, ax):
        X = MinMaxScaler().fit_transform(X)
        labels = np.unique(y).tolist()
        for label in labels:
            ax.scatter(
                X[y.ravel() == label, 0],
                X[y.ravel() == label, 1],
                marker=r"${}$".format(label),
                s=int(300 * fig_size[-1] / 15),
                color=plt.cm.Dark2(labels.index(label)),
                alpha=0.425,
                zorder=2,
            )
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(X.shape[0]):
            # plot every digit on the embedding
            # show an annotation box for a group of digits
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            if photo_dir:
                shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
                image = plt.imread(os.path.join(photo_dir, names[i,0]))
                height, width = image.shape
                ratio = 0.25 * fig_size[-1] / 15
                image = cv2.resize(image, (int(ratio*width), int(ratio*height)))
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(image, cmap="gray", clim=[0, 255]),
                    X[i], bboxprops=dict(edgecolor=plt.cm.Dark2(labels.index(y[i])))
                )
                imagebox.set(zorder=1)
                ax.add_artist(imagebox)

        ax.set_title(title)
        ax.axis("off")

    # Fit loop
    from time import time

    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X

        print(f"Computing {name}...")
        start_time = time()
        try:
            projections[name] = transformer.fit_transform(data, y)
            timing[name] = time() - start_time
        except Exception as e:
            print(f"Error computing {name}. Skip. See error below:")
            print(e)

    # Plot projections
    from itertools import zip_longest

    fig, axs = plt.subplots(nrows=round(len(embeddings)/2.0), ncols=2, figsize=fig_size, dpi=300)

    for name, ax in zip_longest(timing, axs.ravel()):
        if name is None:
            ax.axis("off")
            continue
        title = f"{name} (time {timing[name]:.3f}s)"
        plot_embedding(projections[name], title, ax)