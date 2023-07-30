# Wasserstein GANs

This repository contains some code for demonstrating the application of Wasserstein GANs (WGANs). The code is used in the following blog posts:

* [Wasserstein GANs](https://www.fabriziomusacchio.com/blog/2023-07-29-wgan/)
* [Eliminating the middleman: Direct Wasserstein distance computation in WGANs without discriminator](https://www.fabriziomusacchio.com/blog/2023-07-30-wgan_with_direct_wasserstein_distance/)
* [Conditional GANs](https://www.fabriziomusacchio.com/blog/2023-07-30-cgan/)

For further details, please refer to these posts.

Results of training a default GAN on the MNIST dataset for 50 epochs:
![gif](GAN_images/depp_conv_gan.gif)

Results of training a Wasserstein GAN on the same dataset:
![gif](WGAN_images/depp_conv_wgan.gif)

Results of training a Wasserstein GAN using the POT library, avoiding the necessity of a discriminator:
![gif](GAN_demo_images/cross_animation.gif)


Results of training a conditional GAN:

![gif](cGAN_demo_images/cGAN_animation_edited.gif)


For reproducibility:

```powershell
conda create -n gan -y python=3.9
conda activate gan
conda install mamba -y
mamba install -y numpy matplotlib scikit-learn scipy pot tensorflow imageio pillow ipykernel
mamba install -y pytorch torchvision -c pytorch
pip install POT
```

If you want to run the code on a Mac with Apple Silicon (M1, M2), install tensorflow and pytorch as described here:


* [How to run TensorFlow on the M1 Mac GPU](https://www.fabriziomusacchio.com/blog/2022-11-10-apple_silicon_and_tensorflow/)
* [How to run PyTorch on the M1 Mac GPU](https://www.fabriziomusacchio.com/blog/2022-11-18-apple_silicon_and_pytorch/)





