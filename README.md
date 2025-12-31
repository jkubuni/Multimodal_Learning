# CILP Multimodal Learning Assessment

This project implements and analyzes multimodal learning architectures and CLIP training and evaluation. It focuses on combining LiDAR and RGB data using various fusion strategies. The project includes:
- Data exploration and visualization using FiftyOne (`notebooks/01_dataset_exploration.ipynb`).
- A comparative study of Late vs. Intermediate Fusion architectures (`notebooks/02_fusion_comparison.ipynb`).
- An ablation study comparing MaxPool2d and Strided Convolution for downsampling (`notebooks/03_strided_conv_ablation.ipynb`).
- A final assessment of multimodal projection capabilities (`notebooks/04_final_assessment.ipynb`).
- Experiment tracking using Weights & Biases (Check out the logs and graphs at the link below).

**Weights & Biases Username:** jan-kubeler-hpi

## Links & External Resources

- **Public W&B Project Link:** https://wandb.ai/jan-kubeler-hpi/clip-extended-assessment/
- **Google Drive with Code, Data and Checkpoints:** https://drive.google.com/drive/folders/1urKJ8vRw5ysw9J5p5BVaF6CMD2FLpg2A?usp=sharing

## Setup & Environment

**Environment Specifications:**
- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (Recommended)

**Setup Instructions (Local):**
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Setup Instructions (Colab):**
1. Copy (or Link) the drive folder provided above into your own drive.
2. Launch the notebook of interest in Colab.
3. If you changed the path, make sure to change the mount paths/directories where the code is looking for the `data` and `src` directories. (Usually the first cell.) 

**Data Setup Instructions:**
Ensure the data is placed in a `data/` folder in the project root with the following structure:
```
data/
    cubes/
        azimuth.npy
        zenith.npy
        lidar/
            ...
        rgb/
            ...
    spheres/
        azimuth.npy
        zenith.npy
        lidar/
        rgb/
```

## Summary of Results

**Final Metrics Summary (Task 5):**

|                     | Validation Loss | Projector MSE | Classifier Accuracy |
|---------------------|-----------------|---------------|---------------------|
| CLIP Assessment     | 0.16092         | 0.75988       | 98.59375            |

**Fusion Comparison (Task 3):**
There is not really a noticeable difference between the three architectures in this specific task. All four models achive up to 100% accuracy. This is most likely due to the very simple nature of the task. The most relevant difference are the parameter counts, which greatly influence the training time. 

**Ablation Study Summary (Task 4):**
The differences here are also mostly negligable. The strided convolutions seems to perform slightly better, but in a task as simple as this one, the differences are really hard to pinpoint.

## Reproducibility & Technical Details

**Instructions to Reproduce:**
To replicate the results, execute the notebooks in the `notebooks/` directory in the following order:
1. `notebooks/01_dataset_exploration.ipynb`: For initial data analysis.
2. `notebooks/02_fusion_comparison.ipynb`: To run the fusion strategy comparison.
3. `notebooks/03_strided_conv_ablation.ipynb`: To run the ablation study.
4. `notebooks/04_final_assessment.ipynb`: To train and evaluate the final model.

**Repository Structure:**
- `src/`: Contains the core Python modules:
  - `datasets.py`: Data loading and preprocessing logic.
  - `models.py`: Model architecture definitions (encoders, fusion modules, etc.).
  - `training.py`: Training loops and validation functions.
  - `visualization.py`: Helper functions for plotting and FiftyOne integration.
- `notebooks/`: Jupyter notebooks for each task of the assessment.
- `checkpoints/`: Directory where model checkpoints are saved.
- `results/`: Directory for saving assessment results and visualizations.
- `data/`: Dataset directory (not included in repo, must be set up locally).

**Random Seed Management:**
Random seeds are set at the beginning of training scripts/notebooks to ensure reproducibility. In my experiments I used a seed of 51.