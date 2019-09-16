import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE


def last_layer_display(last_layer, test_y):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    last_layer = last_layer.cpu()  # shape [100, 1568=32 * 7 * 7] 一幅图1568个特征
    test_y = test_y.cpu()  # shape [100]
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy())  # shape [100, 2]
    labels = test_y.numpy()
    plot_with_labels(low_dim_embs, labels)


def plot_with_labels(low_d_weights, labels):
    plt.cla()
    x_t, y_t = low_d_weights[:, 0], low_d_weights[:, 1]
    for x, y, s in zip(x_t, y_t, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(x_t.min(), x_t.max())
    plt.ylim(y_t.min(), y_t.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


def get_digit_picture(data, label):
    plt.imshow(data.numpy(), cmap='gray')
    plt.title('%i' % label)
    plt.pause(1)

