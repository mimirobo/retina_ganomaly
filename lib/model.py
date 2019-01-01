"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

##
class Ganomaly:
    """GANomaly Class
    """

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'Ganomaly'

    def __init__(self, opt, dataloader=None):
        super(Ganomaly, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.gpu_ids!=-1 else "cpu")

        # GDrive Parameters
        # self.gdrive_netg_created = False
        # self.gdrive_netd_created = False
        # self.gdrive_loss_created = False

        # -- Discriminator attributes.
        self.out_d_real = None
        self.feat_real = None
        self.err_d_real = None
        self.fake = None
        self.latent_i = None
        self.latent_o = None
        self.out_d_fake = None
        self.feat_fake = None
        self.err_d_fake = None
        self.err_d = None

        # -- Generator attributes.
        self.out_g = None
        self.err_g_bce = None
        self.err_g_l1l = None
        self.err_g_enc = None
        self.err_g = None

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        # print(self.netg)
        # print(self.netd)

        ##
        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.input.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])

    ##
    def update_netd(self):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        # BCE
        self.netd.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.netd(self.input)
        self.err_d_real = self.bce_criterion(self.out_d_real, self.label)
        self.err_d_real.backward()
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

        self.out_d_fake, self.feat_fake = self.netd(self.fake.detach())
        self.err_d_fake = self.bce_criterion(self.out_d_fake, self.label)
        # --
        self.err_d_fake.backward()
        self.err_d = self.err_d_real + self.err_d_fake
        self.optimizer_d.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        #print('Reloading d net')

    ##
    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(z)))  + ||G(z) - x||           #
        # ============================================================ #

        """
        self.netg.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_g, _ = self.netd(self.fake)

        self.err_g_bce = self.bce_criterion(self.out_g, self.label)
        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)  # constrain x' to look like x
        self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i)
        self.err_g = self.err_g_bce * self.opt.w_bce + self.err_g_l1l * self.opt.w_rec + self.err_g_enc * self.opt.w_enc

        self.err_g.backward(retain_graph=True)
        self.optimizer_g.step()

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        self.update_netd()
        self.update_netg()

        # If D loss is zero, then re-initialize netD
        if self.err_d_real.item() < 1e-5 or self.err_d_fake.item() < 1e-5:
            self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([('err_d', self.err_d.item()),
                              ('err_g', self.err_g.item()),
                              ('err_d_real', self.err_d_real.item()),
                              ('err_d_fake', self.err_d_fake.item()),
                              ('err_g_bce', self.err_g_bce.item()),
                              ('err_g_l1l', self.err_g_l1l.item()),
                              ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))
        #upload the saved weights to google drive
        # self.upload_weights_to_google_drive(weight_dir)

    ##
    # def upload_weights_to_google_drive(self, weight_dir):
        # Create NetG First Time
        # if not self.gdrive_netg_created:
        #     self.uploaded_netG = self.drive.CreateFile({'title': 'netG.pth'})
        #     self.gdrive_netg_created = True
        # Update NetG
        # self.uploaded_netG.SetContentFile('{}/netG.pth'.format(weight_dir))
        # self.uploaded_netG.Upload()
        # print('Uploaded netG file with ID {}'.format(self.uploaded_netG.get('id')))

        # Create NetD First Time
        # if not self.gdrive_netd_created:
        #     self.uploaded_netD = self.drive.CreateFile({'title': 'netD.pth'})
        #     self.gdrive_netd_created = True
        # Update NetD
        # self.uploaded_netD.SetContentFile('{}/netD.pth'.format(weight_dir))
        # self.uploaded_netD.Upload()
        # print('Uploaded netD file with ID {}'.format(self.uploaded_netD.get('id')))

        # Create LossLog First Time
        # if not self.gdrive_loss_created:
        #     self.uploaded_log = self.drive.CreateFile({'title': 'loss_log.txt'})
        #     self.gdrive_loss_created = True
        # Update LossLog
        # self.uploaded_log.SetContentFile('{}/../../loss_log.txt'.format(weight_dir))
        # self.uploaded_log.Upload()
        # print('Uploaded log file with ID {}'.format(self.uploaded_log.get('id')))

    def train_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, disable=True, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name(), self.epoch+1, self.opt.niter))
        self.visualizer.print_current_errors(self.epoch, errors)
    ##
    def train(self):

        # """Create Google PyDrive Client
        # """
        # auth.authenticate_user()
        # gauth = GoogleAuth()
        # gauth.credentials = GoogleCredentials.get_application_default()
        # self.drive = GoogleDrive(gauth)


        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        self.best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_epoch()
            res = self.test()
            if res['AUC'] > self.best_auc:
                self.best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, self.best_auc)
        print(">> Training model %s.[Done]" % self.name())

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name().lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name())
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, self.opt, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            #if(auc >= self.best_auc):
            #    torch.set_printoptions(profile="full")
            #    print('\n\n*****************\nScores:\n{}\n*****************\n'.format(self.an_scores))
            #    print('\n\n*****************\nGT Labels:\n{}\n*****************\n'.format(self.gt_labels))

            # if auc >= self.best_auc:
            #     torch.set_printoptions(profile="full")
            #     print("\nAUC:{}".format(auc))
            #     print('\n\n*****************\nScores:\n{}\n*****************\n'.format(self.an_scores))
            #     print('\n\n*****************\nGT Labels:\n{}\n*****************\n'.format(self.gt_labels))
            if self.opt.phase == 'test':
                torch.set_printoptions(profile="full")
                print("\nAUC:{}".format(auc))
                print('\n\n*****************\nScores:\n{}\n*****************\n'.format(self.an_scores))
                print('\n\n*****************\nGT Labels:\n{}\n*****************\n'.format(self.gt_labels))
            # else:
            #     if(auc >= self.best_auc):
            #         torch.set_printoptions(profile="full")
            #         print('\n\n*****************\nScores:\n{}\n*****************\n'.format(self.an_scores))
            #         print('\n\n*****************\nGT Labels:\n{}\n*****************\n'.format(self.gt_labels))

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance
