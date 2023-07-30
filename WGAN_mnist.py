"""
A script to train a Wasserstein GAN on the MNIST dataset. It's a modification of the
GAN_mnist.py script. The main difference is in the loss functions and the optimizers.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: July 28, 2023
"""
# %% IMPORTS
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import os
# check whether WGAN_images folder is already there, otherwise create it:
if not os.path.exists('WGAN_images'):
    os.makedirs('WGAN_images')
# %% LOAD DATA
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# %% GENERATOR
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# %% DISCRIMINATOR
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # Linear output

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)
# %% DEFINE LOSS FUNCTIONS AND OPTIMIZERS FOR BOTH MODELS
# define the optimizer for both models:
generator_optimizer = tf.keras.optimizers.RMSprop(0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(0.00005)

# define the loss functions:
"""
The basic idea behind the Wasserstein loss is to compute the Wasserstein distance 
between the real and generated distributions. The discriminator in a WGAN
is trained to estimate this distance. The loss function for the discriminator is defined 
as the difference of the average discriminator's scores on real and generated samples. 
That's the reason why we compute real_loss as -tf.reduce_mean(real_output) and 
fake_loss as tf.reduce_mean(fake_output). The overall discriminator's loss is then
real_loss + fake_loss.

For the generator, we want to minimize the opposite of the average discriminator's 
score on the generated samples, so the generator's loss is -tf.reduce_mean(fake_output).
"""
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return real_loss + fake_loss

def gradient_penalty(real_images, fake_images):
    alpha = tf.random.uniform(shape=[real_images.shape[0], 1, 1, 1], minval=0., maxval=1.)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp

# Save checkpoints
checkpoint_dir = './training_checkpoints_wgan'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# %% MAIN LOOP FUNCTIONS
# We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF):
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gp = gradient_penalty(images, generated_images)
        disc_loss += gp
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
  gen_losses = []  # List to store generator losses
  disc_losses = []  # List to store discriminator losses

  for epoch in range(epochs):
    start = time.time()

    gen_loss_avg = tf.keras.metrics.Mean()  # Metric to compute average generator loss
    disc_loss_avg = tf.keras.metrics.Mean()  # Metric to compute average discriminator loss

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_avg(gen_loss)  # Add current generator loss
      disc_loss_avg(disc_loss)  # Add current discriminator loss

    gen_losses.append(gen_loss_avg.result())  # Append average generator loss for this epoch
    disc_losses.append(disc_loss_avg.result())  # Append average discriminator loss for this epoch

    # produce images for the GIF as you go:
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)

    # save the model every 15 epochs:
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # generate after the final epoch:
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)

  return gen_losses, disc_losses

def generate_and_save_images(model, epoch, test_input):
    # Note, that `training` is set to False. This is so all layers run in 
    # inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Epoch: {epoch}", fontsize=16)
    plt.savefig('WGAN_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
# %% TRAINING
gen_losses, disc_losses = train(train_dataset, EPOCHS)
# %% CREATE GIF
# restore the latest checkpoint:
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# display a single image using the epoch number:
def display_image(epoch_no):
  return PIL.Image.open('WGAN_images/image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)

anim_file = 'WGAN_images/dcwgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('WGAN_images/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
  
# plot Generator and Discriminator Loss:
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_losses, label="Generator")
plt.plot(disc_losses, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('WGAN_images/losses.png')
plt.show()
# %% END
