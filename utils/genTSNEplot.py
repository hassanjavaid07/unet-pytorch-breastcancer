import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


# Make t-SNE plot for the encoded data
def genTSNEplot(encoder, subset_data):
    encoded_data = [encoder(image.unsqueeze(0)).detach().numpy().flatten() for image, _, _ in subset_data]
    encoded_data = np.array(encoded_data)

    labels = [label for _, _, label in subset_data]

    tsne = TSNE(n_components=2, random_state=21)
    tsne_data = tsne.fit_transform(encoded_data)

    # Plot t-SNE embeddings
    colors = ['#A52A2A', '#40E0D0', '#FF00FF']
    cmap = mcolors.ListedColormap(colors)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], marker='.', c=labels, cmap=cmap)
    plt.title('t-SNE Plot of Encoder Output')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    cbar = plt.colorbar(label='Breastcancer Class')
    cbar.set_ticks([0, 1, 2])
    plt.show()
