import os
os.getcwd()

from PIL import Image
import numpy as np
im = Image.open('./imgs/0001.jpg')
np.asarray(im.convert('L'))