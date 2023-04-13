import numpy as np
from PIL import Image
import joblib
from skimage.transform import resize

def load_objects():
    pca = joblib.load('models/pca.joblib')
    model = joblib.load('models/model.joblib')
    return pca, model

def read_image(path_image):
    img = np.asarray(Image.open(path_image).convert('L'))
    img2 = resize(img, output_shape = (28,28), anti_aliasing = False)
    img3 = (img2 - img2.min()) / (img2.max() - img2.min())
    img4 = img3.reshape(1,img3.shape[0]*img3.shape[1])
    return img4

def make_prediction(path_image):
    pca, model = load_objects()
    img = read_image(path_image)
    img_pca = pca.transform(img)
    return model.predict(img_pca)
