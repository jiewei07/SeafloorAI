# SeafloorAI: The First Large-Scale AI-Ready Dataset for Seafloor Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue)](https://github.com/YourUsername/SeafloorAI)
[![Paper](https://img.shields.io/badge/NeurIPS-2024-red)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/274de7d60333c0848f42e18ae97f13e3-Abstract-Datasets_and_Benchmarks_Track.html)

**SeafloorAI** is the first extensive AI-ready dataset for seafloor mapping across 5 geological layers, curated in collaboration with marine scientists.

## 🌊 Abstract

A major obstacle to the advancements of machine learning models in marine science, particularly in sonar imagery analysis, is the scarcity of AI-ready datasets. **SeafloorAI** bridges this gap by providing a standardized, large-scale benchmark for semantic segmentation in underwater environments.

## 📊 Dataset Overview

- **Scale:** 62 geo-distributed survey campaigns covering 17,300 km².
- **Size:** 696K sonar images paired with 827K annotated segmentation masks.
- **Resolution:** 224 × 224 per image.
- **Regions:** 9 major geological regions.

### Data Layers
The dataset includes 9 layers:
1. **Raw Signals:** Backscatter, Bathymetry, Slope, Rugosity.
2. **Annotations:** Sediment, Physiographic Zone, Habitat, Fault, Fold.

## 📂 Dataset Structure

The dataset is organized by region and layer types:

```text
SeafloorAI/
├── region1/
│   ├── input/      (Raw sonar images)
│   ├── pzone/      (Physiographic Zone masks)
│   └── sed/        (Sediment masks)
├── region2/
│   ├── input/
│
```

## 🖼️ Samples
## 💻 Visualization & Dataloader

### Simple Visualization
The following script demonstrates how to load and inspect a sample sonar image pair:

```python
import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_sample(region, image_id, layer='sed'):
    """
    Args:
        region (str): Region name (e.g., 'region1')
        image_id (str): Image filename without extension
        layer (str): Target layer (e.g., 'sed', 'pzone')
    """
    # Define paths (Adjust base path if necessary)
    base_path = f"./{region}"
    img_path = os.path.join(base_path, 'input', f"{image_id}.png")
    mask_path = os.path.join(base_path, layer, f"{image_id}.png")

    # Check if files exist
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Error: File not found at {img_path}")
        return

    # Load images
    img = Image.open(img_path).convert('L')  # Grayscale for sonar
    mask = Image.open(mask_path)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Sonar Input ({image_id})")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='jet') 
    ax[1].set_title(f"Segmentation Mask ({layer})")
    ax[1].axis('off')

    plt.show()

# --- Run the visualization ---
if __name__ == "__main__":
    # Example usage:
    # visualize_sample(region='region1', image_id='0001', layer='sed')
    print("Function loaded. Please call visualize_sample with valid paths.")
```

### PyTorch Dataset Integration
For deep learning workflows, we provide a standard `Dataset` implementation in `seafloor_dataset.py`.

The following example demonstrates how to import and use it with a PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader
from seafloor_dataset import SeafloorDataset
import torchvision.transforms as T

# 1. Initialize Dataset
dataset = SeafloorDataset(
    root_dir='./data', 
    region='region1', 
    target='sed',
    transform=T.ToTensor()
)

# 2. Create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

## 📜 Citation

If you use the **SeafloorAI** dataset in your research, please cite the following paper:

```bibtex
@inproceedings{nguyen2024seafloorai,
  title={SeafloorAI: A Large-scale Vision-Language Dataset for Seafloor Geological Survey}, 
  author={Kien X. Nguyen and Fengchun Qiao and Arthur Trembanis and Xi Peng},
  booktitle={Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2024}
}
```
## 📧 Contact & Acknowledgments
We would like to acknowledge the support from USGS and NOAA for providing the raw survey data.

For questions regarding the dataset, please open an issue in this repository.
