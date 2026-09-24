from pathlib import Path

training: Path = Path(__file__).absolute().parent.parent

project: Path = training.parent

data: Path = project/'data'
GAN_out: Path = data/'GAN_output'
original_gan_out: Path = GAN_out/'ReMWGAN_FM'
dynamic_gan_out: Path = GAN_out/'ReMWGAN_FM_Dynamic'
dpl_tmg_gan_out: Path = GAN_out/'DPL_ReMWGAN_FM'
datasets: Path = data/'datasets'
logs: Path = data/'logs'
classifier: Path = data/'classifier_output'

for i in list(vars().values()):
    if isinstance(i, Path):
        i.mkdir(parents=True, exist_ok=True)
        