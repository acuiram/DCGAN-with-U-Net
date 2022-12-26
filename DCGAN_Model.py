import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

### Define the function for downsampling ###
def downsample(filters, size):
    # Initialize the weights
    initializer = tf.random_normal_initializer(0., 0.02)

    down = tf.keras.Sequential()
    # Add the convolutional layer
    down.add(tf.keras.layers.Conv2D(filters, size,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    # Add the activation function
    down.add(tf.keras.layers.LeakyReLU())
    return down

### Define the function for upsampling ###
def upsample(filters, size, apply_dropout=False):
    # Initialize the weights
    initializer = tf.random_normal_initializer(0., 0.02)

    up = tf.keras.Sequential()
    # Add the transposed convolution
    up.add(tf.keras.layers.Conv2DTranspose(filters, size,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False))

    # Add a dropout layer
    if apply_dropout:
        up.add(tf.keras.layers.Dropout(0.2))
    # Add the activation function
    up.add(tf.keras.layers.ReLU())
    return up


def Generator():
    # Input layer
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    ### Encoder network ###
    down_stack = [downsample(64, 4),
                  downsample(128, 4),
                  downsample(256, 4),
                  downsample(512, 4),
                  ]

    ### Decoder network ###
    up_stack = [upsample(512, 4, apply_dropout=True),
                upsample(256, 4),
                upsample(128, 4),
                upsample(64, 4),
                ]

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # Concatenate the layers
        x = tf.keras.layers.Concatenate()([x, skip])

    initializer = tf.random_normal_initializer(0., 0.02)
    # The output layer with 3 color channels
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


### Built the discriminator(discriminator) ###
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    # Input layer
    inputs = tf.keras.Input(shape=(256, 256, 3))

    x = downsample(64, 4)(inputs)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = downsample(512, 4)(x)

    # Add a flatten layer
    x = tf.keras.layers.Flatten()(x)
    # Add a dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)

    # Add sigmoid for binary classification
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs, x)

### Create a class for the GAN ###
class GAN(tf.keras.Model):
    def __init__(self,
                 discriminator,
                 generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, discriminator_optimizer, generator_optimizer, loss_fn):
        super(GAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def train_step(self, zip_data):

        # Unzip the two datasets
        damaged, original = zip_data
        batch_size = tf.shape(original)[0]

        # Train with the real images and the generated images
        generated = self.generator(damaged)
        # Concatenate the images
        combined = tf.concat([generated, original], axis=0)
        # Make labels for the real and fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Get the dicriminator's output
            prediction = self.discriminator(combined)
            # Determine the loss
            d_loss = self.loss_fn(labels, prediction)
        # Calculate the discriminator's gradients
        gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        # Labels for the generator training
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            # Get the output for the damaged image
            prediction = self.discriminator(self.generator(damaged))
            # Determine the generator's loss
            g_loss = self.loss_fn(misleading_labels, prediction)
        # Compute the generator's gradients
        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Update the losses
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"discriminator_loss": self.d_loss_metric.result(), "generator_loss": self.g_loss_metric.result()}

def main():
    ### Initialize the data ###
    damaged_directory = r"C:\Licenta\GAN_IMAGES\damaged\train"
    original_directory = r"C:\Licenta\GAN_IMAGES\original\train"

    real_dataset = tf.keras.utils.image_dataset_from_directory(original_directory,
                                                               label_mode=None,
                                                               batch_size=32,
                                                               image_size=(256, 256),
                                                               shuffle=False)
    damaged_dataset = tf.keras.utils.image_dataset_from_directory(damaged_directory,
                                                                  label_mode=None,
                                                                  batch_size=32,
                                                                  image_size=(256, 256),
                                                                  shuffle=False)

    ### Define a normalization layer ###
    normalization_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)

    ### Normalize the datasets ###
    real_dataset = real_dataset.map(lambda x: normalization_layer(x))
    damaged_dataset = damaged_dataset.map(lambda x: normalization_layer(x))

    n_epochs = 60

    ### Initialise the two networks ###
    discriminator = Discriminator()
    generator = Generator()

    ### Choose the optimisers for both generator and discriminator ###
    LR = 0.0001
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
    generator_optimizer = keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)

    # discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)
    # generator_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)


    img = tf.keras.utils.image_dataset_from_directory(r'C:\Licenta\images\IMAGES\test',
                                                      label_mode=None,
                                                      batch_size=1,
                                                      image_size=(256, 256),
                                                      shuffle=False)

    img = img.map(lambda x: normalization_layer(x))


    class GANMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for img1 in img:
                print(type(img1))
                generated_image = generator(img1)

            generated_image = (generated_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)
            generator.save_weights(r"C:\Licenta\images\new_saved_model\weights_batchnorm\generator_model_weights_"+str(epoch)+".h5")
            generator.save(r"C:\Licenta\images\new_saved_model\model_batchnorm\generator_model_"+str(epoch)+".h5")
            tf.keras.preprocessing.image.save_img("C:\Licenta\images\saved_batchnorm/"+str(epoch)+".jpg", generated_image)
            # plt.figure(epoch)
            # plt.imshow(generated_image)
            # plt.show()
            # generated_image.save(r"C:\Licenta\images\saved_images\generated_image_{epoch}.jpg".format(epoch=epoch))

    gan = GAN(discriminator=discriminator,
              generator=generator)

    gan.compile(
        discriminator_optimizer=discriminator_optimizer,
        generator_optimizer=generator_optimizer,
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )


    gan.fit(tf.data.Dataset.zip((damaged_dataset, real_dataset)),
                epochs=n_epochs,
                verbose=1,
                callbacks=GANMonitor())

    return

if __name__ == '__main__':
     main()
