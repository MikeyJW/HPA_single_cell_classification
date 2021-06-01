'''Variables for src scripts'''

NUC_MODEL = '../models/nuclei-model.pth'

CELL_MODEL = '../models/cell-model.pth'

SEG_CHANNELS = ['red', 'yellow', 'blue', 'green']

CHANNELS = ['red', 'green', 'blue', 'yellow']

RANDOM_STATE = 42

MEAN_CHANNEL_VALUES = [0.485, 0.456, 0.406]

CHANNEL_STD_DEV = [0.229, 0.224, 0.225]

IMG_SIZE = (260, 260)

NUM_CLASSES = 19
