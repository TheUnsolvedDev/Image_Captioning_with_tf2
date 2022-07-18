import tensorflow as tf
import numpy as np

from param import *


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(
        filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def model(input_size=IMAGE_SIZE):
    inputs = tf.keras.layers.Input(input_size)
    x = tf.keras.layers.Lambda(lambda x:x/255)(inputs)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 5, 4]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    num_words = VOCAB
    state_size = STATE_SIZE
    embedding_size = EMBEDDING_SIZE

    decoder_transfer_map = tf.keras.layers.Dense(state_size,
                                                 activation='tanh',
                                                 name='decoder_transfer_map')(x)
    decoder_input = tf.keras.layers.Input(shape=(None, ), name='decoder_input')
    decoder_embedding = tf.keras.layers.Embedding(input_dim=num_words,
                                                  output_dim=embedding_size,
                                                  name='decoder_embedding')(decoder_input)
    decoder_gru1 = tf.keras.layers.GRU(state_size, name='decoder_gru1',
                                       return_sequences=True)([decoder_embedding, decoder_transfer_map])
    decoder_gru2 = tf.keras.layers.GRU(state_size, name='decoder_gru2',
                                       return_sequences=True)([decoder_gru1, decoder_transfer_map])
    decoder_gru3 = tf.keras.layers.GRU(state_size, name='decoder_gru3',
                                       return_sequences=True)([decoder_gru2, decoder_transfer_map])
    outputs = tf.keras.layers.Dense(num_words,
                                    activation='softmax',
                                    name='decoder_output')(decoder_gru3)
    m = tf.keras.models.Model(inputs=[inputs, decoder_input],
                              outputs=outputs)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
    return m


if __name__ == '__main__':
    m = model()
    m.summary()
    # m.save('ker_m.h5')
    # tf.keras.utils.plot_model(m,  show_shapes=True, to_file="model.png")
