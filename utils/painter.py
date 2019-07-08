import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion1.png')

a = [
    [0.95, 0.01, 0.01, 0.01, 0.00, 0.02],
    [0.01, 0.94, 0.05, 0.00, 0.00, 0.00],
    [0.00, 0.01, 0.93, 0.04, 0.01, 0.01],
    [0.00, 0.00, 0.02, 0.97, 0.01, 0.00],
    [0.03, 0.01, 0.05, 0.02, 0.81, 0.07],
    [0.01, 0.00, 0.03, 0.05, 0.07, 0.83]
]

b=[
    [0.52, 0.08, 0.08, 0.16, 0.12, 0.04],
    [0.08, 0.72, 0.00, 0.00, 0.20, 0.00],
    [0.00, 0.04, 0.92, 0.04, 0.00, 0.00],
    [0.00, 0.00, 0.04, 0.96, 0.00, 0.00],
    [0.00, 0.04, 0.00, 0.00, 0.76, 0.20],
    [0.16, 0.04, 0.00, 0.00, 0.28, 0.52]
]

cm = np.array(b)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
ax.figure.colorbar(im, ax=ax)
title = 'HMM'
label_dict = {'bed': 0, 'fall': 1, 'walk' : 2, 'run' : 3, 'sitdown' : 4, 'standup' : 5}
classes = label_dict.keys()
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig('confusion2.png')

