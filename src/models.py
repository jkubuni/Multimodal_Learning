import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    """
    Simple CNN-based embedder for 64x64 images with `in_ch` input channels and `out_dim` output dimensions.
    3 convolutional layers followed by 3 fully connected layers.
    Here we use MaxPooling (`nn.MaxPool2d`) for downsampling.
    """
    def __init__(self, in_ch, out_dim):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv2d(in_ch, 25, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Input 64x64 -> Pool -> 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        self.flattened_size = 100 * 8 * 8
        
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_embs(self, x):
        """
        Get intermediate embeddings before the final fully connected layer.
        Used for contrastive training.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class EmbedderStrided(nn.Module):
    """
    Simple CNN-based embedder for 64x64 images with `in_ch` input channels and `out_dim` output dimensions.
    3 convolutional layers followed by 3 fully connected layers.
    Here we use strided convolutions for downsampling.
    """
    def __init__(self, in_ch, out_dim):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv2d(in_ch, 25, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        # Since we are using learnable convolutions, we need a seperate downsampler for each layer
        self.down1 = nn.Conv2d(25, 25, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1)
        
        self.flattened_size = 100 * 8 * 8
        
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = self.down1(F.relu(self.conv1(x)))
        x = self.down2(F.relu(self.conv2(x)))
        x = self.down3(F.relu(self.conv3(x)))
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_embs(self, x):
        x = self.down1(F.relu(self.conv1(x)))
        x = self.down2(F.relu(self.conv2(x)))
        x = self.down3(F.relu(self.conv3(x)))
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class LateFusionModel(nn.Module):
    """
    Late Fusion Model with separate Embedders for RGB and LiDAR data.
    The embeddings are concatenated and passed to a classifier.
    2 classes: cubes, spheres.
    """
    def __init__(self):
        super().__init__()
        # Separate Embedders for RGB (4 channels) and LiDAR (4 channels)
        self.rgb_Embedder = Embedder(in_ch=4, out_dim=100)
        self.lidar_Embedder = Embedder(in_ch=4, out_dim=100)
        
        # Concatenate embeddings (100+100=200) -> Classifier
        self.classifier = nn.Sequential(
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 2) # 2 classes: cubes, spheres
        )

    def forward(self, rgb, lidar):
        rgb_emb = self.rgb_Embedder(rgb)
        lidar_emb = self.lidar_Embedder(lidar)
        
        combined = torch.cat((rgb_emb, lidar_emb), dim=1)
        output = self.classifier(combined)
        return output

class IntermediateFusionModel(nn.Module):
    """
    Intermediate Fusion Model with fusion after 3 convolutional layers for RGB and LiDAR data.
    Fusion can be 'concat', 'add', or 'multiply'.
    Followed by shared fully connected layers.
    Here we use MaxPooling for downsampling.
    2 classes: cubes, spheres.
    """
    def __init__(self, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        kernel_size = 3
        
        # RGB Stream (4 channels)
        self.rgb_conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.rgb_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.rgb_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        # LiDAR Stream (4 channels)
        self.lidar_conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.lidar_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.lidar_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        self.pool = nn.MaxPool2d(2)

        # Input size depends on fusion type
        if fusion_type == 'concat':
            in_channels = 200 # 100 + 100
        else: # add or multiply
            in_channels = 100
            
        self.fc1 = nn.Linear(in_channels * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, rgb, lidar):
        # RGB Stream
        x_rgb = self.pool(F.relu(self.rgb_conv1(rgb)))
        x_rgb = self.pool(F.relu(self.rgb_conv2(x_rgb)))
        x_rgb = self.pool(F.relu(self.rgb_conv3(x_rgb)))
        
        # LiDAR Stream
        x_lidar = self.pool(F.relu(self.lidar_conv1(lidar)))
        x_lidar = self.pool(F.relu(self.lidar_conv2(x_lidar)))
        x_lidar = self.pool(F.relu(self.lidar_conv3(x_lidar)))
        
        # Fusion
        if self.fusion_type == 'concat':
            x = torch.cat((x_rgb, x_lidar), dim=1)
        elif self.fusion_type == 'add':
            x = x_rgb + x_lidar
        elif self.fusion_type == 'multiply':
            x = x_rgb * x_lidar
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
        # Shared Layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class IntermediateFusionModelStrided(nn.Module):
    """
    Intermediate Fusion Model with fusion after 3 convolutional layers for RGB and LiDAR data.
    Fusion can be 'concat', 'add', or 'multiply'.
    Followed by shared fully connected layers.
    Here we use strided convolutions for downsampling.
    2 classes: cubes, spheres.
    """
    def __init__(self, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        kernel_size = 3
        
        # RGB Stream
        self.rgb_conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.rgb_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.rgb_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        # LiDAR Stream
        self.lidar_conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.lidar_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.lidar_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        # RGB Downsamplers
        self.rgb_down1 = nn.Conv2d(25, 25, 3, 2, 1)
        self.rgb_down2 = nn.Conv2d(50, 50, 3, 2, 1)
        self.rgb_down3 = nn.Conv2d(100, 100, 3, 2, 1)
        # LiDAR Downsamplers
        self.lidar_down1 = nn.Conv2d(25, 25, 3, 2, 1)
        self.lidar_down2 = nn.Conv2d(50, 50, 3, 2, 1)
        self.lidar_down3 = nn.Conv2d(100, 100, 3, 2, 1)
        
        # Input size depends on fusion type
        if fusion_type == 'concat':
            in_channels = 200 # 100 + 100
        else: # add or multiply
            in_channels = 100
            
        self.fc1 = nn.Linear(in_channels * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, rgb, lidar):
        # RGB Stream
        x_rgb = self.rgb_down1(F.relu(self.rgb_conv1(rgb)))
        x_rgb = self.rgb_down2(F.relu(self.rgb_conv2(x_rgb)))
        x_rgb = self.rgb_down3(F.relu(self.rgb_conv3(x_rgb)))
        
        # LiDAR Stream
        x_lidar = self.lidar_down1(F.relu(self.lidar_conv1(lidar)))
        x_lidar = self.lidar_down2(F.relu(self.lidar_conv2(x_lidar)))
        x_lidar = self.lidar_down3(F.relu(self.lidar_conv3(x_lidar)))
        
        # Fusion
        if self.fusion_type == 'concat':
            x = torch.cat((x_rgb, x_lidar), dim=1)
        elif self.fusion_type == 'add':
            x = x_rgb + x_lidar
        elif self.fusion_type == 'multiply':
            x = x_rgb * x_lidar
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContrastivePretraining(nn.Module):
    """
    Contrastive Pretraining Model for RGB and LiDAR embeddings.
    """
    def __init__(self, embedding_size=200, embedder_type='strided'):
        super().__init__()
        if embedder_type == 'strided':
            self.img_embedder = EmbedderStrided(in_ch=4, out_dim=embedding_size)
            self.lidar_embedder = EmbedderStrided(in_ch=4, out_dim=embedding_size)
        else:
            self.img_embedder = Embedder(in_ch=4, out_dim=embedding_size)
            self.lidar_embedder = Embedder(in_ch=4, out_dim=embedding_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, rgb, lidar):
        img_emb = self.img_embedder(rgb)
        lidar_emb = self.lidar_embedder(lidar)
        
        # Normalize embeddings
        img_emb = F.normalize(img_emb, dim=1)
        lidar_emb = F.normalize(lidar_emb, dim=1)
        
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_img = logit_scale * img_emb @ lidar_emb.t()
        logits_per_lidar = logits_per_img.t()
        
        return logits_per_img, logits_per_lidar

class Projector(nn.Module):
    """
    Simple Feedforward Network to project RGB embeddings to LiDAR embedding space.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class RGB2LiDARClassifier(nn.Module):
    """
    Classifier that maps RGB images to LiDAR classification space using a CILP image encoder and a projector.
    """
    def __init__(self, cilp_img_enc, projector, lidar_clf):
        super().__init__()
        self.img_Embedder = cilp_img_enc
        self.projector = projector
        self.lidar_clf_fc3 = lidar_clf.fc3
        
    def forward(self, x):
        x = self.img_Embedder(x)
        x = self.projector(x)
        x = self.lidar_clf_fc3(x)
        return x
