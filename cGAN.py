"""
A script to demonstrate conditional GANs on the MNIST dataset.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: July 30, 2023

ACKNOWLEDGEMENTS:
The base of this code comes from the following Keras tutorial:
https://keras.io/examples/generative/conditional_gan/
"""
# %% IMPORTS
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
if not os.path.exists('cGAN_demo_images'):
    os.makedirs('cGAN_demo_images')
# %% CONSTANTS AND HYPERPARAMETERS
batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128
# %% LOADING THE MNIST DATASET AND PREPROCESSING IT
# we use all the available examples from both the training and test sets:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels:
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# create tf.data.Dataset:
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")
# %% CALCULATING THE NUMBER OF INPUT CHANNELS FOR THE GENERATOR AND DISCRIMINATOR
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)
# %% CREATING THE DISCRIMINATOR AND GENERATOR
# create the discriminator:
discriminator = keras.Sequential(
    [keras.layers.InputLayer((28, 28, discriminator_in_channels)),
     layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
     layers.LeakyReLU(alpha=0.2),
     layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
     layers.LeakyReLU(alpha=0.2),
     layers.GlobalMaxPooling2D(),
     layers.Dense(1)],
    name="discriminator")

# create the generator:
generator = keras.Sequential(
    [keras.layers.InputLayer((generator_in_channels,)),
     # we want to generate 128 + num_classes coefficients to reshape into a
     # 7x7x(128 + num_classes) map:
     layers.Dense(7 * 7 * generator_in_channels),
     layers.LeakyReLU(alpha=0.2),
     layers.Reshape((7, 7, generator_in_channels)),
     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
     layers.LeakyReLU(alpha=0.2),
     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
     layers.LeakyReLU(alpha=0.2),
     layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")],
    name="generator")
# %% CREATING A CONDITIONAL GAN MODEL
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # unpack the data:
        real_images, one_hot_labels = data

        # add dummy dimensions to the labels so that they can be concatenated with
        # the images:
        # this is for the discriminator:
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size])
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes))

        # sample random points in the latent space and concatenate the labels:
        # this is for the generator:
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1)

        # decode the noise (guided by labels) to fake images:
        generated_images = self.generator(random_vector_labels)

        # combine them with real images. Note that we are concatenating the labels
        # with these images here:
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0)

        # assemble labels discriminating real from fake images:
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # train the discriminator:
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))

        # sample random points in the latent space:
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1)

        # assemble labels that say "all real images":
        misleading_labels = tf.zeros((batch_size, 1))

        # train the generator:
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # monitor loss:
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()}
# %% TRAINING THE CGAN
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))

cond_gan.fit(dataset, epochs=20)

# save the model weights for later use:
cond_gan.save_weights('cGAN/cGAN_model_weights_MNIST')
# %% RELOAD THE MODEL WEIGHTS
"""
Only perform this step if you want to reload the model weights.
The model weights need to be saved in a first run.
"""

# initialize your model structure:
model = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
model.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
# load the previously saved weights:
model.load_weights('cGAN/cGAN_model_weights_MNIST')

# extract the trained generator from our Conditional GAN:
trained_gen = model.generator
# %% Interpolating between classes with the trained generator
# We first extract the trained generator from our Conditional GAN.
trained_gen = cond_gan.generator

# choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 50

# sample noise for the interpolation:
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))

def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label)

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake

start_class = 6
end_class = 1

fake_images = interpolate_class(start_class, end_class)

fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("cGAN_demo_images/cGAN_animation.gif", converted_images[:,:,:,0], dpi=200)

# plot and save the first and last frame:
plt.figure(figsize=(5, 5))
plt.imshow(converted_images[10, :, :, 0], cmap='gray')
plt.axis('off')
plt.title(f"generate `{start_class}`", fontsize=22)
plt.tight_layout()
plt.savefig('cGAN_demo_images/cGAN_first_frame.png', dpi=200)
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(converted_images[-1, :, :, 0], cmap='gray')
plt.axis('off')
plt.title(f"generate `{end_class}`", fontsize=22)
plt.tight_layout()
plt.savefig('cGAN_demo_images/cGAN_last_frame.png', dpi=200)
plt.show()
# %% END

