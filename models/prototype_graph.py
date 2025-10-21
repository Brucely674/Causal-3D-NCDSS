"""
Prototype Graph Convolution for Novel Class Discovery
Implements graph-based prototype learning for better class separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PrototypeGraphNCD(nn.Module):
    """
    Prototype Graph Convolution module for Novel Class Discovery
    
    This module implements a graph-based approach to prototype learning that:
    1. Builds a graph connecting points to prototypes
    2. Uses graph convolution to refine features and prototypes
    3. Generates pseudo-labels for unlabeled points
    4. Maintains consistency between known and novel class prototypes
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        num_known_classes: int, 
        num_novel_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        """
        Initialize the Prototype Graph NCD module
        
        Args:
            feature_dim: Dimension of input features
            num_known_classes: Number of known classes
            num_novel_classes: Number of novel classes to discover
            hidden_dim: Hidden dimension for graph convolution layers
            num_layers: Number of graph convolution layers
            temperature: Temperature parameter for attention computation
            dropout: Dropout rate
        """
        super(PrototypeGraphNCD, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_known_classes = num_known_classes
        self.num_novel_classes = num_novel_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Graph convolution layers for feature refinement
        self.feature_layers = nn.ModuleList([
            nn.Linear(feature_dim if i == 0 else hidden_dim, 
                     hidden_dim if i < num_layers - 1 else feature_dim)
            for i in range(num_layers)
        ])
        
        # Prototype refinement layers
        self.prototype_refine_known = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.prototype_refine_novel = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Attention mechanism for prototype selection
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classifiers for known and novel classes
        self.known_classifier = nn.Linear(feature_dim, num_known_classes)
        self.novel_classifier = nn.Linear(feature_dim, num_novel_classes)
        
        # Learnable temperature parameter
        self.temp_param = nn.Parameter(torch.tensor(temperature))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def build_prototype_graph(
        self, 
        point_features: torch.Tensor, 
        known_prototypes: torch.Tensor, 
        novel_prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build graph connections between points and prototypes
        
        Args:
            point_features: Point features [N, feature_dim]
            known_prototypes: Known class prototypes [num_known, feature_dim]
            novel_prototypes: Novel class prototypes [num_novel, feature_dim]
            
        Returns:
            Tuple of (adjacency_matrix, point_prototype_similarity, prototype_weights)
        """
        # Compute similarities between points and prototypes
        all_prototypes = torch.cat([known_prototypes, novel_prototypes], dim=0)  # [num_total, feature_dim]
        
        # Cosine similarity between points and all prototypes
        point_prototype_sim = F.cosine_similarity(
            point_features.unsqueeze(1),  # [N, 1, feature_dim]
            all_prototypes.unsqueeze(0),  # [1, num_total, feature_dim]
            dim=2
        )  # [N, num_total]
        
        # Apply temperature scaling
        scaled_sim = point_prototype_sim / self.temp_param
        
        # Create adjacency matrix (top-k connections)
        k = min(5, all_prototypes.size(0))  # Connect to top-5 most similar prototypes
        _, top_k_indices = torch.topk(scaled_sim, k, dim=1)  # [N, k]
        
        # Create sparse adjacency matrix
        N = point_features.size(0)
        num_total_prototypes = all_prototypes.size(0)
        adjacency_matrix = torch.zeros(N, num_total_prototypes, device=point_features.device)
        
        # Set connections
        for i in range(N):
            for j in range(k):
                prototype_idx = top_k_indices[i, j]
                adjacency_matrix[i, prototype_idx] = scaled_sim[i, prototype_idx]
        
        # Normalize adjacency matrix
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        adjacency_matrix = adjacency_matrix / row_sums
        
        # Compute prototype weights based on point connections
        prototype_weights = adjacency_matrix.sum(dim=0)  # [num_total]
        prototype_weights = F.softmax(prototype_weights, dim=0)
        
        return adjacency_matrix, point_prototype_sim, prototype_weights
    
    def graph_convolution(
        self, 
        point_features: torch.Tensor, 
        prototypes: torch.Tensor, 
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform graph convolution to refine point features
        
        Args:
            point_features: Point features [N, feature_dim]
            prototypes: All prototypes [num_total, feature_dim]
            adjacency_matrix: Adjacency matrix [N, num_total]
            
        Returns:
            Refined point features [N, feature_dim]
        """
        # Apply graph convolution layers
        x = point_features
        
        for i, layer in enumerate(self.feature_layers):
            # Graph convolution: aggregate information from connected prototypes
            prototype_contrib = torch.matmul(adjacency_matrix, prototypes)  # [N, feature_dim]
            
            # Combine point features with prototype information
            if i == 0:
                x = x + prototype_contrib  # Residual connection
            else:
                x = x + self.dropout(prototype_contrib)
            
            # Apply linear transformation
            x = layer(x)
            
            # Apply activation (except for last layer)
            if i < len(self.feature_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x
    
    def refine_prototypes(
        self, 
        point_features: torch.Tensor, 
        known_prototypes: torch.Tensor, 
        novel_prototypes: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine prototypes based on connected points
        
        Args:
            point_features: Point features [N, feature_dim]
            known_prototypes: Known class prototypes [num_known, feature_dim]
            novel_prototypes: Novel class prototypes [num_novel, feature_dim]
            adjacency_matrix: Adjacency matrix [N, num_total]
            
        Returns:
            Tuple of (refined_known_prototypes, refined_novel_prototypes)
        """
        num_known = known_prototypes.size(0)
        num_novel = novel_prototypes.size(0)
        
        # Split adjacency matrix for known and novel prototypes
        known_adj = adjacency_matrix[:, :num_known]  # [N, num_known]
        novel_adj = adjacency_matrix[:, num_known:]  # [N, num_novel]
        
        # Compute weighted point features for each prototype
        known_weighted_features = torch.matmul(known_adj.T, point_features)  # [num_known, feature_dim]
        novel_weighted_features = torch.matmul(novel_adj.T, point_features)  # [num_novel, feature_dim]
        
        # Normalize by connection weights
        known_weights = known_adj.sum(dim=0, keepdim=True).T  # [num_known, 1]
        novel_weights = novel_adj.sum(dim=0, keepdim=True).T  # [num_novel, 1]
        
        known_weights = torch.where(known_weights > 0, known_weights, torch.ones_like(known_weights))
        novel_weights = torch.where(novel_weights > 0, novel_weights, torch.ones_like(novel_weights))
        
        known_weighted_features = known_weighted_features / known_weights
        novel_weighted_features = novel_weighted_features / novel_weights
        
        # Refine prototypes using MLP
        refined_known = known_prototypes + self.prototype_refine_known(known_weighted_features)
        refined_novel = novel_prototypes + self.prototype_refine_novel(novel_weighted_features)
        
        return refined_known, refined_novel
    
    def generate_pseudo_labels(
        self, 
        refined_features: torch.Tensor, 
        refined_known: torch.Tensor, 
        refined_novel: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate pseudo-labels for unlabeled points
        
        Args:
            refined_features: Refined point features [N, feature_dim]
            refined_known: Refined known prototypes [num_known, feature_dim]
            refined_novel: Refined novel prototypes [num_novel, feature_dim]
            
        Returns:
            Pseudo-labels [N] (indices of assigned classes)
        """
        # Compute similarities to refined prototypes
        known_sim = F.cosine_similarity(
            refined_features.unsqueeze(1),  # [N, 1, feature_dim]
            refined_known.unsqueeze(0),     # [1, num_known, feature_dim]
            dim=2
        )  # [N, num_known]
        
        novel_sim = F.cosine_similarity(
            refined_features.unsqueeze(1),  # [N, 1, feature_dim]
            refined_novel.unsqueeze(0),     # [1, num_novel, feature_dim]
            dim=2
        )  # [N, num_novel]
        
        # Combine similarities
        all_sim = torch.cat([known_sim, novel_sim], dim=1)  # [N, num_known + num_novel]
        
        # Generate pseudo-labels (assign to most similar prototype)
        pseudo_labels = torch.argmax(all_sim, dim=1)  # [N]
        
        return pseudo_labels
    
    def forward(
        self, 
        point_features: torch.Tensor, 
        known_prototypes: torch.Tensor, 
        novel_prototypes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the prototype graph convolution
        
        Args:
            point_features: Point features [N, feature_dim]
            known_prototypes: Known class prototypes [num_known, feature_dim]
            novel_prototypes: Novel class prototypes [num_novel, feature_dim]
            
        Returns:
            Dictionary containing:
                - refined_features: Refined point features
                - refined_known: Refined known prototypes
                - refined_novel: Refined novel prototypes
                - known_logits: Logits for known classes
                - novel_logits: Logits for novel classes
                - pseudo_labels: Generated pseudo-labels
                - adjacency_matrix: Graph adjacency matrix
        """
        # Build prototype graph
        adjacency_matrix, point_prototype_sim, prototype_weights = self.build_prototype_graph(
            point_features, known_prototypes, novel_prototypes
        )
        
        # Apply graph convolution to refine features
        refined_features = self.graph_convolution(
            point_features, 
            torch.cat([known_prototypes, novel_prototypes], dim=0), 
            adjacency_matrix
        )
        
        # Refine prototypes
        refined_known, refined_novel = self.refine_prototypes(
            refined_features, known_prototypes, novel_prototypes, adjacency_matrix
        )
        
        # Generate logits
        known_logits = self.known_classifier(refined_features)  # [N, num_known]
        novel_logits = self.novel_classifier(refined_features)  # [N, num_novel]
        
        # Generate pseudo-labels
        pseudo_labels = self.generate_pseudo_labels(refined_features, refined_known, refined_novel)
        
        return {
            'refined_features': refined_features,
            'refined_known': refined_known,
            'refined_novel': refined_novel,
            'known_logits': known_logits,
            'novel_logits': novel_logits,
            'pseudo_labels': pseudo_labels,
            'adjacency_matrix': adjacency_matrix,
            'prototype_weights': prototype_weights
        }


class PrototypeGraphLoss(nn.Module):
    """
    Loss function for prototype graph learning
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.01):
        """
        Initialize the loss function
        
        Args:
            alpha: Weight for classification loss
            beta: Weight for prototype consistency loss
            gamma: Weight for graph regularization loss
        """
        super(PrototypeGraphLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(
        self, 
        results: Dict[str, torch.Tensor], 
        targets: torch.Tensor,
        known_prototypes: torch.Tensor,
        novel_prototypes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss
        
        Args:
            results: Output from PrototypeGraphNCD
            targets: Ground truth labels [N]
            known_prototypes: Original known prototypes
            novel_prototypes: Original novel prototypes
            
        Returns:
            Dictionary containing individual loss components
        """
        # Classification loss
        known_logits = results['known_logits']
        novel_logits = results['novel_logits']
        
        # Combine logits for all classes
        all_logits = torch.cat([known_logits, novel_logits], dim=1)
        classification_loss = F.cross_entropy(all_logits, targets)
        
        # Prototype consistency loss (keep prototypes close to original)
        known_consistency = F.mse_loss(results['refined_known'], known_prototypes)
        novel_consistency = F.mse_loss(results['refined_novel'], novel_prototypes)
        consistency_loss = known_consistency + novel_consistency
        
        # Graph regularization loss (encourage smooth features)
        adjacency_matrix = results['adjacency_matrix']
        refined_features = results['refined_features']
        
        # Laplacian regularization
        degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))
        laplacian = degree_matrix - adjacency_matrix
        graph_loss = torch.trace(torch.matmul(refined_features.T, torch.matmul(laplacian, refined_features)))
        
        # Total loss
        total_loss = (self.alpha * classification_loss + 
                     self.beta * consistency_loss + 
                     self.gamma * graph_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'consistency_loss': consistency_loss,
            'graph_loss': graph_loss
        }
