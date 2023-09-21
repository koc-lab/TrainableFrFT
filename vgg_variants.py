# %%
from configurations.configs import PoolType
from utils.utils import change_pooling
from utils.utils import dump_model_variants_dict

# TODO: Just a single run of this file enough, for all models configure dicts with same context below and dump them to pickle file with dump_model_variants_dict() function

MP = PoolType.MaxPool
FFT = PoolType.FFTPool
DFRFT = PoolType.DFrFTPool
FRFT = PoolType.FrFTPool

CLASS_NAMES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# TODO: ilk 3 MP istediğin gibi değiştirince patlamıyor

VGG_11 = [64, MP, 128, MP, 256, 256, MP, 512, 512, MP, 512, 512, MP]
VGG_11_FFT = [64, MP, 128, MP, 256, 256, FFT, 512, 512, MP, 512, 512, MP]
VGG_11_DFRFT = [64, MP, 128, MP, 256, 256, DFRFT, 512, 512, MP, 512, 512, MP]
VGG_11_FRFT = [64, MP, 128, MP, 256, 256, FRFT, 512, 512, MP, 512, 512, MP]


VGG_13 = [64, 64, MP, 128, 128, MP, 256, 256, MP, 512, 512, MP, 512, 512, MP]
VGG_13_FFT = [64, 64, MP, 128, 128, MP, 256, 256, FFT, 512, 512, MP, 512, 512, MP]
VGG_13_FRFT = [64, 64, MP, 128, 128, MP, 256, 256, FRFT, 512, 512, MP, 512, 512, MP]
VGG_13_DFRFT = [64, 64, MP, 128, 128, MP, 256, 256, DFRFT, 512, 512, MP, 512, 512, MP]



VGG_16_1 = [64, 64, MP, 128, 128, MP, 256, 256, 256, MP]
VGG_16_2 = [512, 512, 512, MP, 512, 512, 512, MP]
VGG_16 = VGG_16_1 + VGG_16_2
VGG_16_FFT = [64, 64, MP, 128, 128, MP, 256, 256, 256, FFT] + VGG_16_2
VGG_16_FRFT = [64, 64, MP, 128, 128, MP, 256, 256, 256, FRFT] + VGG_16_2
VGG_16_DFRFT = [64, 64, MP, 128, 128, MP, 256, 256, 256, DFRFT] + VGG_16_2


VGG_LAYER_DICT = {
    "VGG_11": VGG_11,
    "VGG_11_FFT": VGG_11_FFT,
    "VGG_11_FRFT": VGG_11_FRFT,
    "VGG_11_DFRFT": VGG_11_DFRFT,
    "VGG_13":VGG_13,
    "VGG_13_FFT":VGG_13_FFT,
    "VGG_13_FRFT":VGG_13_FRFT,
    "VGG_13_DFRFT":VGG_13_DFRFT,
    "VGG_16": VGG_16,
    "VGG_16_FFT": VGG_16_FFT,
    "VGG_16_FRFT": VGG_16_FRFT,
    "VGG_16_DFRFT": VGG_16_DFRFT,
}


# VGG_13 = [64, 64, MP, 128, 128, MP, 256, 256, MP, 512, 512, MP, 512, 512, MP]


# VGG_19_1 = [64, 64, MP, 128, 128, MP, 256, 256, 256, 256]
# VGG_19_2 = [MP, 512, 512, 512, 512, MP, 512, 512, 512, 512, MP]
# VGG_19 = VGG_19_1 + VGG_19_2

# asdasd
# VGG_LAYER_DICT = {
#     "VGG_11": VGG_11,
#     "VGG_13": VGG_13,
#     "VGG_16": VGG_16,
#     "VGG_19": VGG_19,
#     "VGG_11_FFT": VGG_11_FFT,
#     "VGG_13_FFT": VGG_13_FFT,
#     "VGG_16_FFT": VGG_16_FFT,
#     "VGG_19_FFT": VGG_19_FFT,
#     "VGG_11_FRFT": VGG_11_FRFT,
#     "VGG_13_FRFT": VGG_13_FRFT,
#     "VGG_16_FRFT": VGG_16_FRFT,
#     "VGG_19_FRFT": VGG_19_FRFT,
#     "VGG_11_DFRFT": VGG_11_DFRFT,
#     "VGG_13_DFRFT": VGG_13_DFRFT,
#     "VGG_16_DFRFT": VGG_16_DFRFT,
#     "VGG_19_DFRFT": VGG_19_DFRFT,
# }


dump_model_variants_dict(model_name="vgg", model_variants_dict=VGG_LAYER_DICT)

# %%
