

from training import(
    config,
    datasets,
    models,
)


# We need to import classes from each of these six files

# from training import(
#     classifier,
#     GAN,
#     SNGAN,
#     WGAN,
#     TMG_GAN,
#     logging,
# )

from training.logger import Logger
from training.tmg_gan_reference import OriginalTMGGAN, TMGGAN
from training.remwgan_fm import ReMWGANFM, MYMETHOD
from training.GAN import GAN
from training.WGAN import WGAN
from training.SNGAN import SNGAN
from training.classifier import Classifier
