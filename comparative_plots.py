from glcm_tools import *
from config import distances, angles, standardize_glcm_image, glcm_levels, props  # GLCM parameters (number_of_features = len(distances) * len(angles) * len(props))

imgs = ["fluid.tiff", "good.tiff", "shark-skin.tiff", "tearing.tiff"]
feature_names = get_glcm_feature_names(distances, angles, props)

fig, axs = plt.subplots(3, len(imgs))
line_types = ['-', '--']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for ax_idx, img_path in enumerate(imgs):
    x_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    glcms, bin_image = region_glcm(x_gray, distances, angles, glcm_levels, standardize=standardize_glcm_image)

    features = get_glcm_features(glcms, props, avg_and_range=False)

    for idx, prop in enumerate(props):
        n_angles = len(angles)
        for n in range(n_angles):
            f = features[0, idx*len(distances)*n_angles + n*len(distances):idx*len(distances)*n_angles + (n+1)*len(distances)]
            f_n = feature_names[idx*len(distances)*n_angles + n*len(distances):idx*len(distances)*n_angles + (n+1)*len(distances)]
            axs[0, ax_idx].plot(distances, f,
                                colors[idx%len(colors)] + line_types[n%n_angles],
                                label="%s° %s" % (angles[n]*180/math.pi , prop))

        # avg = features[0, idx*len(distances)*2:idx*len(distances)*2+len(distances)]
        # r = features[0, idx*len(distances)*2+len(distances):idx*len(distances)*2+len(distances)+len(distances)]
        #
        # axs[0, ax_idx].plot(distances, avg, label="Avg %s" % prop)

    axs[0, ax_idx].set_title(img_path.replace(".tiff", ""))
    axs[0, ax_idx].set_ylim([0, 7])
    axs[0, ax_idx].set_xlim([distances[0], distances[-1]])

    axs[1, ax_idx].imshow(bin_image, cmap="gray", clim=(0, glcm_levels-1))
    axs[1, ax_idx].axes.xaxis.set_visible(False)
    axs[1, ax_idx].axes.yaxis.set_visible(False)

    axs[2, ax_idx].imshow(x_gray, cmap="gray", clim=(0, 255))
    axs[2, ax_idx].axes.xaxis.set_visible(False)
    axs[2, ax_idx].axes.yaxis.set_visible(False)

axs[0, 0].legend(loc='upper left', bbox_to_anchor=(-0.7, 1))
plt.show()