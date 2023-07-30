"""
A script to train a GAN on the MNIST dataset. 
The script uses a Deep Convolutional Generative Adversarial Network and the main
code base is from the TensorFlow tutorial on GANs:

https://www.tensorflow.org/tutorials/generative/dcgan

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
# check whether GAN_images folder is already there, otherwise create it:
if not os.path.exists('GAN_images'):
    os.makedirs('GAN_images')
# %% LOAD DATA AND DEFINE MODEL PARAMETERS
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# define the training parameters:
EPOCHS = 50

# batch and shuffle the data:
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# %% GENERATOR
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# %% DISCRIMINATOR
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
# %% DEFINE LOSS FUNCTIONS AND OPTIMIZERS FOR BOTH MODELS
# this method returns a helper function to compute cross entropy loss:
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss:
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# generator loss:
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# save checkpoints:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# %% MAIN LOOP FUNCTIONS
# We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF):
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

gen_losses = []
disc_losses = []
avg_gen_losses_per_epoch = []
avg_disc_losses_per_epoch = []

# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_losses.append(gen_loss)
      disc_losses.append(disc_loss)
      
    # calculate average generator and discriminator loss for the current epoch:
    avg_gen_loss_this_epoch = np.mean(gen_losses)
    avg_disc_loss_this_epoch = np.mean(disc_losses)
    
    # append these averages to our new lists:
    avg_gen_losses_per_epoch.append(avg_gen_loss_this_epoch)
    avg_disc_losses_per_epoch.append(avg_disc_loss_this_epoch)

    # clear the lists for the next epoch:
    gen_losses.clear()
    disc_losses.clear()

    # produce images for the GIF as you go:
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # save the model every 15 epochs:
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # generate after the final epoch:
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
  return avg_gen_losses_per_epoch, avg_disc_losses_per_epoch

def generate_and_save_images(model, epoch, test_input):
  # Note, that `training` is set to False. This is so all layers run in 
  # inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  # annotate the figure with the epoch number
  plt.suptitle(f"Epoch: {epoch}", fontsize=16)
  plt.savefig('GAN_images/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
# %% Train the model
avg_gen_losses_per_epoch, avg_disc_losses_per_epoch = train(train_dataset, EPOCHS)
# %% CREATE GIF
# restore the latest checkpoint:
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# display a single image using the epoch number:
def display_image(epoch_no):
  return PIL.Image.open('GAN_images/image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)

anim_file = 'GAN_images/depp_conv_gan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('GAN_images/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# plot losses:
plt.figure(figsize=(10,5))
plt.title("Average Generator and Discriminator Loss During Training")
plt.plot(avg_gen_losses_per_epoch,label="Generator")
plt.plot(avg_disc_losses_per_epoch,label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('GAN_images/losses.png', dpi=200)
plt.show()
# %% END
