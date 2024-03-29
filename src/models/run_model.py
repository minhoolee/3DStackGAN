from __future__ import print_function

import argparse
import os
import random
import sys
import pprint
import time
import datetime
import dateutil
import dateutil.tz

import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import summary
from tensorboardX import FileWriter
from tqdm import tqdm

from src.data.datasets import Dataset, GANDataGenerator
from src.models.config import cfg, cfg_from_file
from src.models.utils import mkdir_p
from src.models.utils import open_pickle
from src.models.utils import weights_init
from src.models.utils import save_voxels_results, save_model
from src.models.utils import KL_loss
from src.models.utils import compute_discriminator_loss, compute_generator_loss
from src.logging import log_utils


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

log = log_utils.logger(__name__)


class StackedGAN(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.voxels_dir = os.path.join(output_dir, 'Voxels')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.voxels_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from src.models.create_model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        # log.info(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        # log.info(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            log.info('Load from: {}'.format(cfg.NET_G))
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            log.info('Load from: {}'.format(cfg.NET_D))
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from src.models.create_model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        # log.info(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            log.info('Load from: {}'.format(cfg.NET_G))
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            log.info('Load from: {}'.format(cfg.STAGE1_G))
        else:
            log.info("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            log.info('Load from: {}'.format(cfg.NET_D))
        # log.info(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        with torch.no_grad():
            fixed_noise = \
                Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        count = 0
        log.info('Training stage-{}'.format(stage))
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(tqdm(iter(data_loader), leave=False, total=len(data_loader)), 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_voxels_cpu = data['voxel_tensor']
                txt_embeddings = data['raw_embedding']
                real_voxels = Variable(real_voxels_cpu)
                txt_embeddings = Variable(txt_embeddings)
                if cfg.CUDA:
                    real_voxels = real_voxels.cuda()
                    txt_embeddings = txt_embeddings.cuda()

                #######################################################
                # (2) Generate fake voxels
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embeddings, noise)
                _, fake_voxels, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_voxels, fake_voxels,
                                               real_labels, fake_labels,
                                               mu, self.gpus)
                errD.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_voxels,
                                              real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1
                if i % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD.item())
                    summary_D_r = summary.scalar('D_loss_real', errD_real)
                    summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    summary_G = summary.scalar('G_loss', errG.item())
                    summary_KL = summary.scalar('KL_loss', kl_loss.item())

                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_D_r, count)
                    self.summary_writer.add_summary(summary_D_w, count)
                    self.summary_writer.add_summary(summary_D_f, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                    # save the voxels result for each epoch
                    inputs = (txt_embeddings, fixed_noise)
                    lr_fake, fake, _, _ = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_voxels_results(
                        real_voxels_cpu, fake, epoch, self.voxels_dir)
                    if lr_fake is not None:
                        save_voxels_results(
                            None, lr_fake, epoch, self.voxels_dir)
            end_t = time.time()
            log.info('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                     % (epoch, self.max_epoch, i, len(data_loader),
                        errD.item(), errG.item(), kl_loss.item(),
                        errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)

        save_model(netG, netD, self.max_epoch, self.model_dir)

        self.summary_writer.close()

    def sample(self, datapath, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        log.info('Successfully load sentences from: ', datapath)
        log.info('Total number of sentences:', num_embeddings)
        log.info('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            # if count > 3000:
            #     break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embeddings = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embeddings = txt_embeddings.cuda()

            #######################################################
            # (2) Generate fake voxels
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embeddings, noise)
            _, fake_voxels, mu, logvar = \
                nn.parallel.data_parallel(netG, inputs, self.gpus)
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_voxels[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # log.info('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # log.info('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size


def parse_args():
    parser = argparse.ArgumentParser(description='Train a StackedGAN')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--output_dir', dest='output_dir',
                        type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    # if args.data_dir != '':
    #     cfg.DATA_DIR = args.data_dir
    if args.output_dir != '':
        cfg.OUTPUT_DIR = args.output_dir
    # log.info('Using config:')
    # log.info(pprint.pformat(cfg))
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/%s_%s_%s' % \
                 (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        # image_transform = transforms.Compose([
        #     transforms.RandomCrop(cfg.IMSIZE),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset = Dataset(cfg.DATA_DIR, 'train',
        #                   imsize=cfg.IMSIZE,
        #                   transform=image_transform)
        train_embeddings_path = os.path.join(
            cfg.PROCESSED_DATA_DIR, 'processed_captions_train.p')
        train_input_dict = open_pickle(train_embeddings_path)
        dataset = GANDataGenerator(train_input_dict)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = StackedGAN(output_dir)
        algo.train(dataloader, cfg.STAGE)
    # else:
    #     datapath = '%s/metadata/val/val_captions.t7' % (cfg.DATA_DIR)
    #     algo = StackedGAN(output_dir)
    #     algo.sample(datapath, cfg.STAGE)
