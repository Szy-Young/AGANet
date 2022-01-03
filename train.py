#from datagen import Data_Generator
from datagen_aug import Data_Generator
#from cnn_model import CNN_Model
from cnn_model_att import CNN_Model
from datetime import datetime


# Configs and hyper-params
SKT_DIR = '/media/home_bak/swap/BigActionData/skeletons' 
LABEL_DIR = '/media/home_bak/swap/BigActionData/labels' 
IMG_DIR = '/media/home_bak/swap/BigActionData/frames'

SEQ_LEN = 100 
SEQ_STEP = 5
NUM_KP = 10
DIM_KP = 4
NUM_CLASS = 10

EPOCH = 50
EPOCH_SIZE = 100
BATCH_SIZE = 256
WEIGHT_DECAY = 0.0001
BASE_LR = 0.001
LR_DECAY = 0.2
LR_DECAY_FREQ = 1000

now = datetime.now()
LOGDIR = "logs/%d%02d%02d_%02d%02d/" %(
    now.year, now.month, now.day, now.hour, now.minute)
GPU_MEMORY_FRACTION = 1.0


# Construct data generator
dataset = Data_Generator(skt_dir=SKT_DIR,
                         label_dir=LABEL_DIR,
                         img_dir=IMG_DIR,
                         seq_len=SEQ_LEN,
                         seq_step=SEQ_STEP,
                         num_kp=NUM_KP,
                         dim_kp=DIM_KP,
                         num_class=NUM_CLASS)
dataset.create_sets()
#dataset.load_dataset() 

# Construct model and train
model = CNN_Model(dataset=dataset,
                  logdir=LOGDIR,
                  num_kp=NUM_KP,
                  dim_kp=DIM_KP,
                  seq_len=SEQ_LEN,
                  num_class=NUM_CLASS,
                  weight_decay=WEIGHT_DECAY,
                  base_lr=BASE_LR,
                  lr_decay=LR_DECAY,
                  lr_decay_freq=LR_DECAY_FREQ,
                  epoch=EPOCH,
                  epoch_size=EPOCH_SIZE,
                  batch_size=BATCH_SIZE,
                  gpu_memory_fraction=GPU_MEMORY_FRACTION,
                  training=True)
model.build_model()
model.train()
