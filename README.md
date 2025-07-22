# CelebA GAN Project
This repository contains a Generative Adversarial Network (GAN) and diffusion models implementation designed to generate facial images based on the CelebA dataset, conditioned on attribute labels. The project leverages PyTorch for building and training the GAN, with a focus on generating 256x256 images.

## The project includes:

Data Preparation: A custom CelebAHQDataset class to load and preprocess images and attributes from the CelebA dataset.
Model Architecture: An EnhancedGenerator and a Discriminator with conditional input support for 40 attribute dimensions.
Training Pipeline: A training loop with mixed precision training, gradient penalty, and learning rate scheduling.
Evaluation: Visualization of generated samples and loss plots during training.

## Prerequisites
Software Dependencies

Python 3.8+
PyTorch 1.13+ (with CUDA support if using GPU)
torchvision
pandas
numpy
pillow
tqdm
matplotlib
openpyxl (optional, for Excel attribute files)

## Hardware

GPU with at least 12 GB VRAM recommended for batch size 128.
Approximately 20-30 GB of storage for the CelebA dataset and checkpoints.

## Dataset

Download the CelebA dataset (images) and attribute file (list_attr_celeba.csv) from the official CelebA website or a mirror.

## Installation

Clone the repository:
git clone <repository-url>
cd <repository-directory>


Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision pandas numpy pillow tqdm matplotlib openpyxl


Verify CUDA setup (if using GPU):
import torch
print(torch.cuda.is_available())  # Should print True



##  Usage
Running the Training
Execute the main script to start training:
python train_gan_stabilized.py


Batch Size: Set to 128 (adjustable in the code).
Epochs: Configured for 300 epochs.
Checkpoints: Saved every 10 epochs in the gan_checkpoints directory.
Samples: Generated samples are saved in gan_checkpoints/samples every 10 epochs.

## Monitoring

Training progress is displayed using tqdm, showing epoch and iteration details.
Loss values (Generator and Discriminator) are printed per epoch.
Generated sample images are visualized and saved with titles indicating the conditioned attribute.

## Output

Checkpoints: Stored as .pth files (e.g., best_gan_model_YYYYMMDD_HHMMSS.pth).
Loss Plot: Saved as loss_plot_YYYYMMDD_HHMMSS.png in gan_checkpoints.
Samples: Saved as samples_epoch_X_YYYYMMDD_HHMMSS.png in gan_checkpoints/samples.

## Configuration
Key Parameters

batch_size: 128 (adjust in train_and_evaluate()).
num_epochs: 300.
lr: 2e-4 (initial learning rate).
noise_dim: 100 (dimension of input noise).
checkpoint_interval: 10 (epochs between checkpoint saves).

## Adjustments

Modify data_dir and attr_file paths in train_and_evaluate() to match your dataset location.
Adjust batch_size, num_epochs, or lr based on your hardware and training needs.

## Troubleshooting

NaN Losses: If training results in nan values, ensure the dataset is correctly preprocessed (e.g., no corrupted images). Check the updated train_gan_stabilized.py for stability improvements like reduced gradient penalty and clipped outputs.
Memory Issues: Reduce batch_size (e.g., to 64) if you encounter out-of-memory errors.
Performance: Monitor GPU usage and adjust hyperparameters if training is too slow.
