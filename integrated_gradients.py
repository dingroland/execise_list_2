import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from captum.attr import visualization
from lime import lime_image
from skimage.segmentation import mark_boundaries

from dataset import WaterbirdDataset
from model import ConvNextClassifier



def calculate_integrated_gradients(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_class_index: int,
        baseline: torch.Tensor = None,
        num_steps: int = 10,
) -> torch.Tensor:
    """
    Calculate Integrated Gradients for a given input and target class.

    Args:
        model (torch.nn.Module): The neural network model.
        input_tensor (torch.Tensor): The input tensor to analyze.
        target_class_index (int): The index of the target class.
        baseline (torch.Tensor, optional): The baseline input. If None, uses zero tensor.
        num_steps (int, optional): The number of steps for approximating the integral.

    Returns:
        tuple[torch.Tensor, torch.Tensor, float]: A tuple containing:
            - integrated_gradients: The calculated integrated gradients.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Prepare steps for integrated gradients
    steps = torch.stack([baseline + (i / num_steps) * (input_tensor - baseline) for i in range(num_steps + 1)])
    steps.requires_grad = True

    # Calculate integrated gradients
    activations = model(steps)
    target_activations = activations[:, target_class_index]
    target_activations.sum().backward()
    integrated_grads = (input_tensor - baseline) * torch.mean(steps.grad, dim=0)

    return integrated_grads


def visualize_integrated_gradients(
        original_image: Image.Image,
        integrated_grads: torch.Tensor,
        predicted_class: str,
        true_class: str,
        quantile_threshold: float = 0.04
) -> None:
    """
    Visualize the integrated gradients.

    Args:
        original_image (Image.Image): The original input image.
        integrated_grads (torch.Tensor): The calculated integrated gradients.
        predicted_class (str): The predicted class name.
        true_class (str): The true class name.
        quantile_threshold(float, optional): The percentile threshold to only show top percentile contributions.
    """
    # Convert tensor to numpy array and normalize
    ig_np = integrated_grads.detach().cpu().numpy().transpose(1, 2, 0)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original image
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Plot integrated gradients
    # Apply threshold
    ig_norm = np.sum(ig_np, axis=2)
    threshold_min, threshold_pos = np.quantile(ig_norm, [quantile_threshold / 2, 1 - quantile_threshold / 2])
    ig_norm[np.logical_and((ig_norm > threshold_min), (ig_norm < threshold_pos))] = 0
    ig_norm = np.abs(ig_norm)
    ig_heatmap = axs[1].imshow(ig_norm, cmap="Blues", vmin=ig_norm.min(), vmax=ig_norm.max())
    axs[1].set_title("Integrated Gradients")
    axs[1].axis('off')
    plt.colorbar(ig_heatmap, ax=axs[1], fraction=0.046, pad=0.04)

    # Add overall title
    plt.suptitle(f"Predicted: {predicted_class}, True: {true_class}", fontsize=16)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def main():
    # Load the pre-trained classifier
   
   

    # Define image transformations
    
    

    # Load the dataset
    
    

    # Select a random image (we could also select a specific image)
   
   
    # Get model predictions
    
    
    # Calculate integrated gradients
    
    
    # Visualize results using our custom function
  
  

    # Comparison with Captum


    ## Try different baselines:
    # blured
    

    # Calculate integrated gradients
 

    # Visualize results using our custom function
 

    # random noise

    # Calculate integrated gradients

    # Visualize results using our custom function
 


if __name__ == '__main__':
    main()
