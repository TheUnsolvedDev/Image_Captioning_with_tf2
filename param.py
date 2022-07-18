import tensorflow as tf

BATCH_SIZE = 16
IMAGE_SIZE = [224,224,3]
FULL_DATA_SIZE = 600000
VOCAB = 50000
STATE_SIZE = 512
EMBEDDING_SIZE = 512
MAX_LENGTH = 20
EPOCHS = 30

def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return inputs
