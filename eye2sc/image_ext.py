import numpy
from keras.preprocessing.image import list_pictures, load_img, img_to_array


def list_pictures_in_multidir(paths):
    fpaths = []
    for path in paths:
        fpaths += list_pictures(path)
    
    #fpaths = fpaths.sort()
    print(fpaths)
       
    return fpaths

def load_imgs_asarray(paths, grayscale=False, target_size=None,
                      dim_ordering='default'):
    arrays = []
    for path in paths:
        img = load_img(path, grayscale, target_size)
        array = img_to_array(img, dim_ordering)
        arrays.append(array)
        # print('.', end='') 
        print(path)
    
    return numpy.asarray(arrays)
