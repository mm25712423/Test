import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ImageFilter
import glob


'''
for dirname, dirnames, filenames in os.walk(os.getcwd()):

    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print (file_path)
'''
'''
for dirname, dirnames, filenames in os.walk(os.getcwd()):

    # print path to all filenames.
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        image = Image.open(file_path)
        image.filter(ImageFilter.EDGE_ENHANCE_MORE).show()
'''

for filename in glob.glob(os.getcwd() + '/*/*.jpg'):
    image = Image.open(filename)
    print (filename)
    image.filter(ImageFilter.EDGE_ENHANCE_MORE).save(filename, 'jpeg')
    #image.save(filename, 'jpeg')
