import os
import pydicom as dicom
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps
from torchvision import transforms as torch_transforms
from multiprocessing import Pool
from functools import partial

def convert_dicom_to_jpg(image, meta_data, path_to_data, transform=None):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        image (str): Name of the image to process
        path_to_data (str): Path to the dicom images
        meta_data (pd.Dataframe): Dataframe containing meta information of BrixIA
        transform (torchvision.Transforms): Torchvision transform object
    """

    dcm = dicom.dcmread(os.path.join(path_to_data, image))
    img_array = dcm.pixel_array

    max_gray = np.max(img_array)

    # Scale 16-bit gray values of dicom images
    if max_gray <= 4095:
        img_array = (img_array/4095*255).astype(np.uint8)
    else:
        img_array = (img_array/65535*255).astype(np.uint8)

    img_pil = Image.fromarray(img_array)

    interpretation = meta_data.loc[image]['PhotometricInterpretation']
    if interpretation == 'MONOCHROME1':
        img_pil = PIL.ImageOps.invert(img_pil)

    image = image.replace('.dcm', '.jpg')

    if transform:
        img_pil = transform(img_pil)

    dest_path = path_to_data.replace('dicom_clean', 'images')
    os.makedirs(dest_path, exist_ok=True)

    img_pil.save(os.path.join(dest_path, image))


if __name__ == '__main__':

    print('Started preprossesing of BrixIA')

    path_to_data = os.path.join('./data/brixia/dicom_clean')
    meta_path = os.path.join('./data/brixia/metadata_global_v2.csv')
    meta_data = pd.read_csv(meta_path, sep=';', dtype={'BrixiaScore': str}, index_col='Filename')

    processes = 14
    size = (320,320)

    transforms = torch_transforms = torch_transforms.Compose([
        torch_transforms.Resize(size)
    ])

    images_list = os.listdir(path_to_data)

    pool = Pool(processes=processes)

    wrapper = partial(convert_dicom_to_jpg, meta_data = meta_data, path_to_data=path_to_data, transform=transforms)

    result = pool.map_async(wrapper, images_list)
    result.get()

    print('Finished preprocessing.')
