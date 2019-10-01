import os
import errno
import numpy as np
import csv
import time
import json
import pickle

from copy import deepcopy
from src.models.config import cfg
from datetime import datetime

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils

from src.data.nrrd_rw import write_nrrd


#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_voxels, fake_voxels,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_voxels.size(0)
    cond = conditions.detach()
    fake = fake_voxels.detach()
    real_features = nn.parallel.data_parallel(netD, (real_voxels), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # real pairs
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_voxels, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_voxels), gpus)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_voxels_results(data_voxels, fake, epoch, voxels_dir):
    num = cfg.VIS_COUNT
    # fake = fake[0:num]
    # data_voxels is changed to [0,1]
    for i in range(num):
        epoch_dir = os.path.join(voxels_dir, 'epoch_%03d' % epoch)
        mkdir_p(epoch_dir)
        fake_voxels = fake[i].data.cpu().numpy()
        if data_voxels is not None:
            # voxels = data_voxels[i].numpy()
            # write_nrrd(voxels, os.path.join(epoch_dir, 'real_sample_%03d.nrrd' % i))
            # vutils.save_image(
            #     data_voxels, '%s/real_samples.png' % image_dir,
            #     normalize=True)
            # fake.data is still [-1, 1]
            write_nrrd(fake_voxels, os.path.join(epoch_dir, 'fake_sample_%03d.nrrd' % i))
            # vutils.save_image(
            #     fake.data, '%s/fake_samples_epoch_%03d.png' %
            #     (image_dir, epoch), normalize=True)
        else:
            write_nrrd(fake_voxels, os.path.join(epoch_dir, 'lr_fake_sample_%03d.nrrd' % i))
            # vutils.save_image(
            #     fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            #     (image_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_%d.pth' % (model_dir, epoch))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# def get_available_devices():
#     """Return a list of gpu devices on the current machine.
#     """
#     from tensorflow.python.client import device_lib
#     return device_lib.list_local_devices()
#
#
# def get_trainable_variables_by_scope(scope):
#     """Gets the trainable variables in the specified scope (including variables
#     such as scope/asdf/var1).
#
#     Args:
#         scope: A string representing the scope.
#
#     Returns
#         scope_vars: A list of trainable variables in the scope.
#     """
#     trainable_vars = tf.trainable_variables()
#     if scope[-1] != '/':
#         sc = scope + '/'
#     else:
#         sc = scope
#     return [var for var in trainable_vars if sc in var.name]


# def get_num_iterations(iters_per_epoch, num_epochs=None, disp=True):
#     """Calculate the number of iterations from the number of epochs and
#     the batch size.
#
#     Args:
#         iters_per_epoch: Number of samples in the dataset.
#         num_epochs: Number of epochs to run for.
#         disp: Boolean for whether to print n_iteration/epoch/batch size info
#
#     Returns:
#         max_steps: Number of iterations necessary to reach the desired number of epochs.
#     """
#     if num_epochs is None:
#         num_epochs = cfg.TRAIN.NUM_EPOCHS
#
#     if iters_per_epoch is None:
#         return None
#
#     max_steps = iters_per_epoch * num_epochs
#
#     # print info
#     if disp:
#         print('--------- NUMBER OF TRAINING STEPS ---------')
#         print('number of epochs:', num_epochs)
#         print('batch size:', cfg.CONST.BATCH_SIZE)
#         print('number of steps per epoch:', iters_per_epoch)
#         print('total number of training steps:', max_steps)
#
#     return max_steps


def sample_z(batch_size, dist=None):
    """Returns a numpy array batch of the sampled noise. Call this to get a noise batch sample.
    """
    if dist is None:
        dist = cfg.GAN.NOISE_DIST

    if dist == 'gaussian':
        return np.random.normal(loc=cfg.GAN.NOISE_MEAN,
                                scale=cfg.GAN.NOISE_STDDEV,
                                size=[batch_size, cfg.GAN.NOISE_SIZE])
    elif dist == 'uniform':
        return np.random.uniform(low=-cfg.GAN.NOISE_UNIF_ABS_MAX,
                                 high=cfg.GAN.NOISE_UNIF_ABS_MAX,
                                 size=[batch_size, cfg.GAN.NOISE_SIZE])
    else:
        raise ValueError('Sample distribution must be uniform or gaussian.')


# def print_tensor_shapes(tensor_list, prefix=''):
#     """Prints the tensor shapes given a list of tensors.
#
#     Args:
#         tensor_list: A list of TensorFlow tensors.
#         prefix: String that will be prepended to what printed string.
#     """
#     for tensor in tensor_list:
#         print('{}{} shape:'.format(prefix, tensor.name),
#               tensor.get_shape())


def undo_rescale_voxel_tensor(rescaled_voxel_tensor):
    """Takes a voxel tensor with values in [-1, 1] and rescales them to [0, 1].

    Args:
        rescaled_voxel_tensor: A single voxel tensor after rescaling (values in [-1, 1]).

    Returns:
        voxel_tensor: A single voxel tensor with values in [0, 1].
    """
    unscaled_voxel_tensor = (rescaled_voxel_tensor + 1.) / 2.
    return unscaled_voxel_tensor


def add_noise_to_text_embedding(text_embedding_batch):
    """Concatenates the text embedding and noise to generate the final input embedding to the
    shape generator.
    """
    assert text_embedding_batch.ndim == 2

    batch_size = text_embedding_batch.shape[0]
    noise_batch = sample_z(batch_size)
    return np.concatenate((text_embedding_batch, noise_batch), axis=1)


# def print_train_step_data(data_dict, step):
#     """Prints the data stored in the dictionary. The print is formatted assuming that the data_dict
#     corresponds to a specific step in a loop.
#
#     Args:
#         data_dict: Dictionary of things to print.
#         step: Step (such as global step) in the loop.
#     """
#     print('----------- train step %06d ---------------' % (step + 1))
#     for key, val in data_dict.items():
#         print_str = '{}\t{}: {}'.format(str(datetime.now()), key, val)
#         tf.logging.info(print_str)


# def compute_sequence_length(input_batch):
#     """Creates ops for computing sequence length given the input batch.
#
#     Modified from the source: https://danijar.com/variable-sequence-lengths-in-tensorflow/
#
#     Args:
#         input_batch: A BxC tensor where B is batch size, C is max caption
#             length. 0 indicates the padding, a non zero positive value indicate a word index
#
#     Returns:
#         seq_length: Tensor of size [batch_size] representing the length of
#             each caption in the current batch.
#     """
#     # used represents a BxC tensor where (i, j) element is 1 if the jth
#     # word of the ith sample (in the batch) is used/present
#     with tf.variable_scope('seq_len'):
#         used = tf.greater(input_batch, 0)
#         seq_length = tf.reduce_sum(tf.cast(used, tf.int32), reduction_indices=1)
#         seq_length = tf.cast(seq_length, tf.int32)
#
#     return seq_length


def augment_voxel_tensor(voxel_tensor, max_noise=10):
    """Augments the RGB values of the voxel tensor. The RGB channels are perturbed by the same
    single noise value, and the noise is sampled from a uniform distribution.

    Args:
        voxel_tensor: A single voxel tensor.
        max_noise: Integer representing max noise range. We will perform voxel_value + max_noise
            to augment the voxel tensor, where voxel_value and max_noise are [0, 255].

    Returns:
        augmented_voxel_tensor: Voxel tensor after the data augmentation.
    """
    augmented_voxel_tensor = np.copy(voxel_tensor)  # Do nothing if binvox
    if (voxel_tensor.ndim == 4) and (voxel_tensor.shape[3] != 1) and (max_noise > 0):
        noise_val = float(np.random.randint(-max_noise,
                                            high=(max_noise + 1))) / 255
        augmented_voxel_tensor[:, :, :, :3] += noise_val
        augmented_voxel_tensor = np.clip(augmented_voxel_tensor, 0., 1.)
    return augmented_voxel_tensor


def rescale_voxel_tensor(voxel_tensor):
    """Rescales all values (RGBA) in the voxel tensor from [0, 1] to [-1, 1].

    Args:
        voxel_tensor: A single voxel tensor.

    Returns:
        rescaled_voxel_tensor: A single voxel tensor after rescaling.
    """
    rescaled_voxel_tensor = voxel_tensor * 2. - 1.
    return rescaled_voxel_tensor


# def write_list_to_txt(data_list, txt_output_path, add_numbers=False):
#     """Writes each element in the data list as a single line in a new txt file.
#
#     Args:
#         data_list: List of data to write to the txt file. Each element will be converted to a str.
#         txt_output_path: Path to the new txt file (e.g. '/tmp/test.txt')
#     """
#     print('Writing to: {}'.format(txt_output_path))
#     with open(txt_output_path, 'w') as f:
#         for idx, data_el in enumerate(data_list):
#             if add_numbers is True:
#                 f.write('%04d  %s\n' % (idx, str(data_el)))
#             else:
#                 f.write('{}\n'.format(data_el))
#     print('Done writing.')
#

def convert_idx_to_words(idx_to_word, data_list):
    """Converts each sentence/caption in the data_list using the idx_to_word dict.

    Args:
        idx_to_word: A dictionary mapping word indices (keys) in string format (?) to words.
        data_list: A list of dictionaries. Each dictionary contains a 'raw_embedding' field (among
            other fields) that is a list of word indices.

    Returns:
        sentences: A list of sentences (strings).
    """
    sentences = []
    for idx, cur_dict in enumerate(data_list):
        sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)]
                                                      for word_idx in cur_dict['raw_caption_embedding']
                                                      if word_idx != 0]))

    return sentences


def print_sentences(json_path, data_list):
    # Opens the processed captions generated from tools/preprocess_captions.py
    inputs_list = json.load(open(json_path, 'r'))
    idx_to_word = inputs_list['idx_to_word']

    if isinstance(data_list, list):
        sentences = convert_idx_to_words(idx_to_word, data_list)
    elif isinstance(data_list, np.ndarray):
        sentences = []
        for idx in range(data_list.shape[0]):
            sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)]
                                                          for word_idx in data_list[idx, :] if word_idx != 0]))

    for sentence in sentences:
        print(sentence + '\n')


def write_sentences_txt(json_path, data_list, save_path):
    # Opens the processed captions generated from tools/preprocess_captions.py
    inputs_list = json.load(open(json_path, 'r'))
    idx_to_word = inputs_list['idx_to_word']

    sentences = convert_idx_to_words(idx_to_word, data_list)

    with open(save_path, 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


# def get_json_path(dataset=None):
#     if dataset is None:
#         dataset = cfg.CONST.DATASET
#     if dataset == 'shapenet':
#         return cfg.DIR.JSON_PATH
#     elif dataset == 'primitives':
#         return cfg.DIR.PRIMITIVES_JSON_PATH
#     else:
#         raise ValueError('Please use a supported dataset (shapenet, primitives).')


def get_word_idx_mappings():
    """Return idx2word and word2idx.
    """
    json_path = get_json_path()
    inputs_list = json.load(open(json_path, 'r'))
    idx_to_word = inputs_list['idx_to_word']
    word_to_idx = inputs_list['word_to_idx']
    return idx_to_word, word_to_idx


def convert_embedding_list_to_batch(embedding_list, dtype):
    if embedding_list[0] is None:
        return None
    else:
        return np.array(embedding_list).astype(np.float32)


# def get_learned_embedding_shape():
#     """Get the shape of the learned embeddings (not including batch size).
#     """
#     if ((cfg.CONST.DATASET == 'primitives') and (cfg.CONST.SYNTH_EMBEDDING is True)):
#         with open(cfg.DIR.PRIMITIVES_SYN_EMBEDDINGS, 'rb') as f:
#             synth_embeddings = pickle.load(f)
#         embedding_shape = [len(synth_embeddings['box-beige-h100-r100'])]
#         return embedding_shape
#     elif (cfg.CONST.DATASET == 'primitives') and (cfg.CONST.SYNTH_EMBEDDING is False):
#         embedding_data = open_pickle(cfg.DIR.PRIMITIVES_METRIC_EMBEDDINGS_TRAIN)
#         sample_tuple = embedding_data['caption_embedding_tuples'][0]
#         embedding_shape = list(sample_tuple[3].shape)
#         assert len(embedding_shape) == 1
#         return embedding_shape
#     elif cfg.CONST.DATASET == 'shapenet':
#         embedding_data = open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TRAIN)
#         sample_tuple = embedding_data['caption_embedding_tuples'][0]
#         embedding_shape = list(sample_tuple[3].shape)
#         assert len(embedding_shape) == 1
#         return embedding_shape
#     else:
#         print('Dataset:', cfg.CONST.DATASET)
#         print('Synthetic embeddings:', cfg.CONST.SYNTH_EMBEDDING)
#         raise NotImplementedError('Please implement for other datasets.')


def open_pickle(pickle_file):
    """Open a pickle file and return its contents.
    """
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def get_unique_el_mapping(data_list):
    """Map each unique element in the data list to a number/index.
    """
    key2idx = {el: idx for idx, el in enumerate(set(data_list))}
    idx2key = {v: k for k, v in key2idx.items()}

    for idx in idx2key.keys():
        assert isinstance(idx, int)

    return key2idx, idx2key


# def get_attr_from_cat(category, attr_type):
#     """Return the attribute string given the category. This function parses the category string.
#     """
#     cur_shape, cur_color, cur_height, cur_radius = category.split('-')
#     if attr_type == 'shape':
#         attr = cur_shape
#     elif attr_type == 'color':
#         attr = cur_color
#     elif attr_type == 'height':
#         attr = cur_height
#     elif attr_type == 'radius':
#         attr = cur_radius
#     else:
#         raise ValueError('Please enter a valid attribute type.')
#     return attr


# def debug_validation_stuff(save_vars, p_name):
#     var_name_dict = {var.name: var for var in save_vars}
#     var_weight_dict = {}
#     for k, v in var_name_dict.items():
#         cur_var_name = k
#         cur_var_weight = v.eval()
#         var_weight_dict[cur_var_name] = cur_var_weight
#
#     everything_dict = {
#         'var_weight_dict': var_weight_dict,
#     }
#     with open(p_name, 'wb') as f:
#         pickle.dump(everything_dict, f)
#     print('Wrote debug validation pickle file.')
#
#
# def extract_last_output(output, seq_length):
#         batch_size = tf.shape(output)[0]
#         max_length = tf.shape(output)[1]
#         out_size = int(output.get_shape()[2])
#
#         index = tf.range(0, batch_size) * max_length + (seq_length - 1)
#         flat = tf.reshape(output, [-1, out_size])
#         relevant = tf.gather(flat, index)
#         return relevant
#
#
# def change_dataset_size(inputs_dict, new_dataset_size=5):
#     assert len(inputs_dict.keys()) == 3
#
#     orig_caption_tuples = inputs_dict['caption_embedding_tuples']
#     orig_dataset_size = inputs_dict['dataset_size']
#     new_caption_tuples = orig_caption_tuples[:new_dataset_size]
#
#     inputs_dict['orig_caption_embedding_tuples'] = orig_caption_tuples
#     inputs_dict['orig_dataset_size'] = orig_dataset_size
#     inputs_dict['caption_embedding_tuples'] = new_caption_tuples
#     inputs_dict['dataset_size'] = new_dataset_size
#
#     return inputs_dict
#

# class AttributeWriter(object):
#
#     def __init__(self):
#         with open(cfg.DIR.PRIMITIVES_SYN_EMBEDDINGS, 'rb') as f:
#             self.synth_embeddings = pickle.load(f)
#
#         self.first_row = [
#             'id',
#             'box',
#             'cylinder',
#             'pyramid',
#             'cone',
#             'torus',
#             'sphere',
#             'red',
#             'green',
#             'blue',
#             'teal',
#             'beige',
#             'cyan',
#             'pink',
#             'yellow',
#             'orange',
#             'purple',
#             'brown',
#             'black',
#             'white',
#             'gray',
#             'small',
#             'large',
#             'short',
#             'tall',
#             'thin',
#             'wide',
#             'below',
#             'by',
#         ]
#
#     def write_from_categories(self, filenames_list, categories_list, csv_output_path):
#         """Writes the attributes to a CSV file given a list of categories.
#
#         Args:
#             categories_list: List of category names such as
#                     ['box-beige-h100-r100', 'box-beige-h100-r20', 'box-beige-h100-r50', ...]
#             csv_output_path: Path to output csv file (e.g. '/tmp/output.csv')
#         """
#         attributes_list = []
#         for filename, category in zip(filenames_list, categories_list):
#             attributes_list.append([filename] + self.synth_embeddings[category])
#         self.write_from_attributes(attributes_list, csv_output_path)
#
#     def write_from_attributes(self, attributes_list, csv_output_path):
#         """Write a list of attributes (each element in attributes_list is a list of attributes) to
#         a csv file.
#
#         Args:
#             attributes_list: The first element in each element of attributes list should be the
#                     filename.
#         """
#         with open(csv_output_path, 'w', newline='') as f:
#             writer = csv.writer(f, delimiter=',')
#             writer.writerow(self.first_row)
#             for attribute in attributes_list:
#                 writer.writerow(attribute)


class Timer(object):
    """A simple timer.
    """

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
