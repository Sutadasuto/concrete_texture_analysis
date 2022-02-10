import cv2
import matplotlib.pyplot as plt
import os


def save_classification_comparison(clf, X, Y, names, img_dir, results_dir, prob=False, fig_dim=[480, 720]):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if prob:
        try:
            preds = clf.predict_proba(X)
        except:
            prob = False
            preds = clf.predict(X)
        class_names = clf.classes_
    else:
        preds = clf.predict(X)

    for idx, name in enumerate(names[:, 0]):
        label = Y[idx, 0]
        pred = preds[idx]

        img = plt.imread(os.path.join(img_dir, name))
        dpi = 300
        im_height, im_width = img.shape
        fig_height, fig_width = fig_dim

        # What size does the figure need to be in inches to fit the image?
        fig_size = fig_width / float(dpi), fig_height / float(dpi)
        im_size = (im_width / fig_width), (im_height / fig_height)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes([(1 - im_size[0]) / 2, (1 - im_size[1]) / 2, im_size[0], im_size[1]])
        ax.axis('off')

        # Display the image.
        plt.imshow(img, cmap="gray", clim=(0, 255))
        plt.title("%s - %s" % (label, name), fontdict={'size': 6})
        if prob:
            probs = []
            for j, class_name in enumerate(class_names):
                probs.append("%s: %s" % (class_name, round(pred[j], 2)))
            plt.text(5, 10, ", ".join(probs), fontsize=4, bbox={'facecolor': 'white', 'pad': 5})
        else:
            plt.text(5, 5, pred, fontsize=4, bbox={'facecolor': 'white', 'pad': 5})
        plt.savefig(os.path.join(results_dir, "%s_%s" % (label, name)))
        plt.clf()


def save_keras_classification_comparison(clf, X, Y, names, img_dir, results_dir, one_hot_encoder, fig_dim=[480, 720]):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    preds = clf.predict(X)
    Y = one_hot_encoder.inverse_transform(Y)
    class_names = one_hot_encoder.categories_[0]

    for idx, name in enumerate(names[:, 0]):
        label = Y[idx, 0]
        pred = preds[idx]

        img = plt.imread(os.path.join(img_dir, name))
        dpi = 300
        im_height, im_width = img.shape
        fig_height, fig_width = fig_dim

        # What size does the figure need to be in inches to fit the image?
        fig_size = fig_width / float(dpi), fig_height / float(dpi)
        im_size = (im_width / fig_width), (im_height / fig_height)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes([(1-im_size[0])/2, (1-im_size[1])/2, im_size[0], im_size[1]])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Display the image.
        plt.imshow(img, cmap="gray", clim=(0, 255))

        plt.title("%s - %s" % (label, name), fontdict={'size': 6})
        probs = []
        for j, class_name in enumerate(class_names):
            probs.append("%s: %s" % (class_name, round(pred[j], 2)))
        plt.text(5, 10, ", ".join(probs), fontsize=4, bbox={'facecolor': 'white', 'pad': 2})
        plt.savefig(os.path.join(results_dir, "%s_%s" % (label, name)))
        plt.close(fig)