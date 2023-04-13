import numpy as np
from PIL import Image
import joblib
from skimage.transform import resize

def load_objects():
    pca = joblib.load('models/pca.joblib')
    model = joblib.load('models/model.joblib')
    return pca, model

def read_image(path):
    img = np.asarray(Image.open(path).convert('L'))
    img2 = resize(img, output_shape = (28,28), anti_aliasing = True)
    img3 = (img2 - img2.min()) / (img2.max() - img2.min())
    return img3

def make_prediction():
    pass
