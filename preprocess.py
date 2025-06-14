import scipy.io
import numpy as np

def load_data():
    mat = scipy.io.loadmat('data/wafer_map/LSWMD.mat')
    data = mat['LSWMD'][0]
    images, labels = [], []

    for item in data[:1000]:
        label = int(item[1][0][0])
        wafer_map = item[0][0]
        if wafer_map.size == 0:
            continue
        images.append(wafer_map)
        labels.append(label)

    images = np.array([np.pad(img, ((0, 30 - img.shape[0]), (0, 30 - img.shape[1])), 'constant') for img in images])
    images = np.expand_dims(images, -1)
    return images / 255.0, np.array(labels)