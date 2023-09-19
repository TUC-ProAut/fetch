"""download the files for the TinyImagenet-200 Dataset and convert them into the ImageFolder format. See the main function for conficuration. Based on: https://gist.github.com/lromor/bcfc69dcf31b2f3244358aea10b7a11b#file-tinyimagenet-py-L54"""

# config
DATASETS_ROOT = '/daten/marwei/pytorch'                     # location of all datasets
FILENAME = 'tiny-imagenet-200.zip'                          # name of zip-file
BASE_FOLDER = 'tiny-imagenet-200'                           # name of dir extracted from zip-file
NEW_NAME = 'TinyImagenet'                                   # new name for the ImageFolder dir
URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' 
ZIP_MD5 = '90528d7ca1a48142e341f4ef8d21d0de'


import os
import shutil
from torchvision.datasets.utils import download_and_extract_archive


def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


def main():

    dataset_folder = os.path.join(os.path.expanduser(DATASETS_ROOT), BASE_FOLDER)
    dataset_folder_renamed = os.path.join(os.path.expanduser(DATASETS_ROOT), NEW_NAME)

    if os.path.exists(dataset_folder) or os.path.exists(dataset_folder_renamed):
        raise FileExistsError(f"{dataset_folder} or {os.path.join(os.path.expanduser(DATASETS_ROOT), NEW_NAME)} already exists")

    print('Download')
    download_and_extract_archive(URL, DATASETS_ROOT, filename=FILENAME, remove_finished=True, md5=ZIP_MD5)

    print('restructuring val to ImageFolder')
    normalize_tin_val_folder_structure(os.path.join(dataset_folder, 'val'))

    # GDumb expects the val-dir to be named test
    print('Rename and reorder')
    if os.path.exists(os.path.join(dataset_folder, 'test')) and os.path.exists(os.path.join(dataset_folder, 'val')):
        shutil.rmtree(os.path.join(dataset_folder, 'test'))
        os.rename(os.path.join(dataset_folder, 'val'), os.path.join(dataset_folder, 'test'))
    os.rename(dataset_folder, dataset_folder_renamed)

    print('Done!')
    

if __name__ == '__main__':
    main()
