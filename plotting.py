#!/usr/bin/env python3

import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses, filename="learning_curve.png"):
    """
    Plots the learning curves for training and validation losses and saves the plot to a file.
    
    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        filename (str): The name of the file to save the plot (default is 'learning_curve.png').
    """
    epochs = range(1, len(train_losses) + 1)  # Create an epoch range
    
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend()
    
    # Save the plot to a file
    plt.savefig(filename)
    
    # Show the plot
    plt.show()