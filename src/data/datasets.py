from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd

from src.models.config import cfg
from src.logging import log_utils
from src.data.preprocess import load_voxel
from src.models.utils import augment_voxel_tensor, rescale_voxel_tensor, convert_embedding_list_to_batch

log = log_utils.logger(__name__)


class Dataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        self.split = split
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        metadata_dir = os.path.join(data_dir, 'metadata')
        split_dir = os.path.join(metadata_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(
            split_dir, embedding_type)
        self.class_id = self.load_class_id(
            split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    # def load_bbox(self):
    #     data_dir = self.data_dir
    #     bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    #     df_bounding_boxes = pd.read_csv(bbox_path,
    #                                     delim_whitespace=True,
    #                                     header=None).astype(int)
    #     #
    #     filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    #     df_filenames = \
    #         pd.read_csv(filepath, delim_whitespace=True, header=None)
    #     filenames = df_filenames[1].tolist()
    #     log.info('Total filenames: ', len(filenames), filenames[0])
    #     #
    #     filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    #     numImgs = len(filenames)
    #     for i in xrange(0, numImgs):
    #         # bbox = [x-left, y-top, width, height]
    #         bbox = df_bounding_boxes.iloc[i][1:].tolist()
    #
    #         key = filenames[i][:-4]
    #         filename_bbox[key] = bbox
    #     #
    #     return filename_bbox

    # def load_all_captions(self):
    #     caption_dict = {}
    #     for key in self.filenames:
    #         caption_name = '%s/text/%s.txt' % (self.data_dir, key)
    #         captions = self.load_captions(caption_name)
    #         caption_dict[key] = captions
    #     return caption_dict
    #
    # def load_captions(self, caption_name):
    #     cap_path = caption_name
    #     with open(cap_path, "r") as f:
    #         captions = f.read().decode('utf8').split('\n')
    #     captions = [cap.replace("\ufffd\ufffd", " ")
    #                 for cap in captions if len(cap) > 0]
    #     return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            log.info('embeddings: {}'.format(embeddings.shape))
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')
        log.info('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        # cls_id = self.class_id[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s/%s.jpg' % (data_dir, self.split, key)
        img = self.get_img(img_name, bbox)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)


# class ShapeNetDataset(data.Dataset):
#     def __init__(self, data_dir, metadata_dir, split='train', embedding_type='cnn-rnn',
#                  imsize=64, transform=None, target_transform=None):
#
#         self.transform = transform
#         self.target_transform = target_transform
#         self.imsize = imsize
#         self.data = []
#         self.data_dir = data_dir
#         self.metadata_dir = metadata_dir
#
#         # Need to store list of filepaths
#         # Need to store captions
#         # Need to store embeddings
#
#         # self.filepaths = self.load_filepaths(split_dir)
#         # self.embeddings = self.load_embedding(
#         #     split_dir, embedding_type)
#         # self.captions = self.load_all_captions()
#
#     def load_metadata(self, metadata_dir):
#         """
#         Adds all metadata annotations from CSV files and parses them into the dataset.
#
#         Arguments:
#             metadata_dir (String): directory containing the CSV files to process
#         """
#         # Likely parse a single file into a dict
#         # self.description = ...
#         pass
#
#     # def __load_metadatum(self, metadata_file):
#     #     """
#     #     Adds all metadata annotations from CSV file and parses them into the dataset.
#     #
#     #     Arguments:
#     #         metadata_file (String): CSV file to process
#     #     """
#
#     def clean(self, overwrite=False):
#         """
#         Clean dataset according to rules (should not need to be called repeatedly).
#         """
#         for i in range(__len__()):
#             # ... = __get__item(i)
#             # Determine if it needs to be removed
#             # If needs to be removed, remove from Dataset
#                 # If needs to be overwritten, remove it from data_dir (using filepath)
#             # Else save it to a cache directory
#         pass
#
#     # def __clean(self, index, overwrite=False):
#     #     return False
#
#     def save(self, cache_dir):
#         for i in range(__len__()):
#             # Save each item to the cache_dir
#             pass
#
#     def generate_filepaths(self):
#         # For every directory in data_dir:
#             # For every directory in these directories:
#                 # filepath = os.path.join(data_dir, ...)
#                 # self.filepaths.append(filepath)
#         return None
#
#     def load_filepaths(self, filepaths_file):
#         # self.filepaths = ...
#         pass
#
#     def __getitem__(self, index):
#         # return voxels, embedding
#         return None
#
#     def __len__(self):
#         # return len(self.filepaths)
#         return None

class GANDataGenerator(data.Dataset):
    """Data generator for GAN. Generators single items instead of minibatches.
    """

    def __init__(self, data_dict):
        """Initialize the Data Generator.

        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
        """
        if 'caption_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_tuples']
        elif 'caption_embedding_tuples' in data_dict:
            self.caption_tuples = data_dict['caption_embedding_tuples']
        else:
            raise KeyError('inputs dict does not contain proper keys.')
        self.max_sentence_length = len(self.caption_tuples[0][0])
        self.class_labels = data_dict.get('class_labels')
        problematic_nrrd_path = os.path.join(cfg.PROCESSED_DATA_DIR, 'problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p')
        if problematic_nrrd_path is not None:
            with open(problematic_nrrd_path, 'rb') as f:
                self.bad_model_ids = pickle.load(f)
        else:
            self.bad_model_ids = None
        # super(GANDataGenerator, self).__init__(data_queue=None, data_dict=data_dict, repeat=False)

    def is_bad_model_id(self, model_id):
        """Code reuse.
        """
        if self.bad_model_ids is not None:
            return model_id in self.bad_model_ids
        else:
            return False

    def get_caption_data(self, db_ind):
        """Gets the caption data corresponding to the index specified by db_ind.

        NOTE: Copied directly from GANDataProcess.

        Args:
            db_ind: The integer index corresponding to the index of the caption in the dataset.

        Returns:
            cur_raw_embedding
            cur_category
            cur_model_id
            cur_voxel_tensor
        """
        while True:
            caption_tuple = self.caption_tuples[db_ind]
            cur_raw_embedding = caption_tuple[0].astype(np.float32)
            cur_raw_embedding = cur_raw_embedding / np.linalg.norm(cur_raw_embedding)
            cur_category = caption_tuple[1]
            cur_model_id = caption_tuple[2]

            if self.is_bad_model_id(cur_model_id):
                db_ind = np.random.randint(self.num_data)  # Choose new caption
                continue

            try:
                # cur_learned_embedding = self.get_learned_embedding(caption_tuple)
                cur_voxel_tensor = load_voxel(cur_category, cur_model_id)
                cur_voxel_tensor = augment_voxel_tensor(cur_voxel_tensor,
                                                        # max_noise=cfg.TRAIN.AUGMENT_MAX)
                                                        max_noise=0)
                # Reshape from (H x W x D x C) to (C x H x W x D)
                cur_voxel_tensor = np.transpose(cur_voxel_tensor, (3, 0, 1, 2))
                # if self.class_labels is not None:
                #     cur_class_label = self.class_labels[cur_category]
                # else:
                #     cur_class_label = None

            except FileNotFoundError:  # Retry if we don't have binvoxes
                db_ind = np.random.randint(self.num_data)
                continue
            break
        caption_data = {'raw_embedding': cur_raw_embedding,
                        # 'category': cur_category,
                        # 'model_id': cur_model_id,
                        'voxel_tensor': cur_voxel_tensor}
        return caption_data

    def __getitem__(self, index):
        return self.get_caption_data(index)

    def __len__(self):
        return len(self.caption_tuples)
