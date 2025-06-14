# Update your metrics_visualization.py
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def plot_to_base64(plt):
    """Convert matplotlib plot to base64 encoded image"""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_precision_recall_curve(precision, recall, k_values):
    fig, ax = plt.subplots()
    ax.plot(k_values, precision, label="Precision@K")
    ax.plot(k_values, recall, label="Recall@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall @ K")
    ax.legend()
    ax.grid(True)
    return fig

def plot_ndcg(ndcg_scores, k_values):
    fig, ax = plt.subplots()
    ax.plot(k_values, ndcg_scores, label="NDCG@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("NDCG @ K")
    ax.legend()
    ax.grid(True)
    return fig

def plot_map(map_scores):
    fig, ax = plt.subplots()
    ax.plot(map_scores, label="MAP")
    ax.set_xlabel("Queries")
    ax.set_ylabel("Score")
    ax.set_title("Mean Average Precision (MAP)")
    ax.legend()
    ax.grid(True)
    return fig

def display_metrics_graphs(metrics):
    precision = metrics.get("precision", [])
    recall = metrics.get("recall", [])
    ndcg = metrics.get("ndcg", [])
    map_scores = metrics.get("map", [])
    
    if not precision or not recall:
        return None
    
    k_values = list(range(1, len(precision) + 1))
    
    # Create a combined plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Precision-Recall
    ax1.plot(k_values, precision, label="Precision@K")
    ax1.plot(k_values, recall, label="Recall@K")
    ax1.set_title("Precision & Recall @ K")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: NDCG
    ax2.plot(k_values, ndcg, label="NDCG@K")
    ax2.set_title("NDCG @ K")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: MAP
    if map_scores:
        ax3.plot(range(len(map_scores)), map_scores, label="MAP")
        ax3.set_title("Mean Average Precision (MAP)")
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    return fig