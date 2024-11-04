# main.py

import torch
from dataloader import load_and_preprocess_data, prepare_datasets
from train import train_gan
from testing import evaluate_gan
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Train and Evaluate GAN for Time Series Generation')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to your data folder')
    parser.add_argument('--save_path', type=str, default='./results', help='Path to save results and models')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space for generator')
    parser.add_argument('--generator_model', type=str, default='CNNGenerator',
                        help='Generator model to use: CNNGenerator, LSTMGenerator')
    parser.add_argument('--discriminator_model', type=str, default='CNNDiscriminator',
                        help='Discriminator model to use: CNNDiscriminator, LSTMDiscriminator')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    vS_all = load_and_preprocess_data(args.data_path)
    train_set, test_set, scaler = prepare_datasets(vS_all, batch_size=args.batch_size)

    # Train the GAN with specified models
    generator, discriminator = train_gan(
        train_set,
        device,
        args.save_path,
        nb_epoch=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        generator_model=args.generator_model,
        discriminator_model=args.discriminator_model
    )

    # Evaluate the GAN
    all_real_data = []
    for data in train_set:
        all_real_data.append(data)
    all_real_data = torch.cat(all_real_data)

    # Pass latent_dim to evaluate_gan
    evaluate_gan(generator, all_real_data, device, args.save_path, args.latent_dim)
    print("Training and evaluation completed.")

if __name__ == '__main__':
    main()

