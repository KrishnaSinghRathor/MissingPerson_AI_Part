import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ================= CONFIG =================
DATA_ROOT = "processed_vggface2"
IMG_SIZE = 112
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.1
EMBEDDING_SIZE = 512
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ArcFace parameters
MARGIN = 0.5  # angular margin
SCALE = 64    # feature scale

CHECKPOINT_DIR = "checkpoints2"
OUTPUT_DIR = "training_outputs2"
LOG_INTERVAL = 100
SAVE_INTERVAL = 2  # Save every 2 epochs
# ==========================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"Device: {DEVICE}")
print(f"Data root: {DATA_ROOT}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Output directory: {OUTPUT_DIR}")


# ================= METRICS TRACKER =================
class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.batch_losses = []
        self.batch_accuracies = []
        
    def add_epoch(self, loss, acc, lr, time):
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(time)
    
    def add_batch(self, loss, acc):
        self.batch_losses.append(loss)
        self.batch_accuracies.append(acc)
    
    def save_json(self, path):
        data = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'total_time': sum(self.epoch_times),
            'final_loss': self.train_losses[-1] if self.train_losses else None,
            'final_accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
            'best_accuracy': max(self.train_accuracies) if self.train_accuracies else None
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Metrics saved to {path}")


# ================= DATASET =================
class VGGFace2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Build dataset
        identities = sorted(os.listdir(root_dir))
        for idx, identity in enumerate(identities):
            identity_path = os.path.join(root_dir, identity)
            if not os.path.isdir(identity_path):
                continue
            
            self.class_to_idx[identity] = idx
            
            for img_name in os.listdir(identity_path):
                img_path = os.path.join(identity_path, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((img_path, idx))
        
        self.num_classes = len(self.class_to_idx)
        print(f"Dataset: {len(self.samples)} images, {self.num_classes} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ================= DATA AUGMENTATION =================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ================= MODEL ARCHITECTURE =================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.prelu(out)
        return out


class ResNetFace(nn.Module):
    def __init__(self, num_layers=50, embedding_size=512):
        super(ResNetFace, self).__init__()
        
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]
        elif num_layers == 50:
            layers = [3, 4, 14, 3]
        else:
            raise ValueError(f"Invalid num_layers: {num_layers}")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        
        self.layer1 = self._make_layer(64, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x


# ================= ARCFACE LOSS =================
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = nn.functional.normalize(embeddings, dim=1)
        weight = nn.functional.normalize(self.weight, dim=1)
        
        # Cosine similarity
        cosine = nn.functional.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=DEVICE)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        loss = self.ce(output, labels)
        return loss


# ================= TRAINING =================
def train_epoch(model, criterion, dataloader, optimizer, epoch, metrics_tracker):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            embeddings_norm = nn.functional.normalize(embeddings, dim=1)
            weight_norm = nn.functional.normalize(criterion.weight, dim=1)
            logits = nn.functional.linear(embeddings_norm, weight_norm) * SCALE
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Track batch metrics
        batch_acc = 100. * (predicted == labels).sum().item() / labels.size(0)
        metrics_tracker.add_batch(loss.item(), batch_acc)
        
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ================= VISUALIZATION =================
def plot_training_curves(metrics, save_path):
    """Create comprehensive training visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics - Face Recognition Model', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, len(metrics.train_losses) + 1))
    
    # Plot 1: Loss curve
    axes[0, 0].plot(epochs, metrics.train_losses, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss Over Epochs', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(epochs)
    
    # Annotate final loss
    final_loss = metrics.train_losses[-1]
    axes[0, 0].annotate(f'Final: {final_loss:.4f}', 
                        xy=(epochs[-1], final_loss),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Accuracy curve
    axes[0, 1].plot(epochs, metrics.train_accuracies, marker='s', linewidth=2, markersize=8, color='#2ecc71')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training Accuracy Over Epochs', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(epochs)
    
    # Annotate final accuracy
    final_acc = metrics.train_accuracies[-1]
    axes[0, 1].annotate(f'Final: {final_acc:.2f}%', 
                        xy=(epochs[-1], final_acc),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 3: Learning rate schedule
    axes[1, 0].plot(epochs, metrics.learning_rates, marker='^', linewidth=2, markersize=8, color='#9b59b6')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(epochs)
    
    # Plot 4: Time per epoch
    axes[1, 1].bar(epochs, metrics.epoch_times, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Training Time Per Epoch', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticks(epochs)
    
    # Add total time annotation
    total_time = sum(metrics.epoch_times)
    axes[1, 1].text(0.95, 0.95, f'Total: {total_time/60:.1f} min', 
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                    ha='right', va='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    plt.close()


def plot_loss_accuracy_combined(metrics, save_path):
    """Create combined loss and accuracy plot"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epochs = list(range(1, len(metrics.train_losses) + 1))
    
    # Plot loss on left axis
    color1 = '#e74c3c'
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', color=color1, fontsize=13, fontweight='bold')
    line1 = ax1.plot(epochs, metrics.train_losses, marker='o', linewidth=2.5, 
                     markersize=8, color=color1, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot accuracy on right axis
    ax2 = ax1.twinx()
    color2 = '#2ecc71'
    ax2.set_ylabel('Accuracy (%)', color=color2, fontsize=13, fontweight='bold')
    line2 = ax2.plot(epochs, metrics.train_accuracies, marker='s', linewidth=2.5,
                     markersize=8, color=color2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    plt.title('Training Loss and Accuracy Progress', fontsize=15, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined plot saved to {save_path}")
    plt.close()


def create_summary_table(metrics, num_classes, num_images, save_path):
    """Create a summary table image"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    total_time = sum(metrics.epoch_times)
    avg_time = total_time / len(metrics.epoch_times) if metrics.epoch_times else 0
    final_loss = metrics.train_losses[-1] if metrics.train_losses else 0
    final_acc = metrics.train_accuracies[-1] if metrics.train_accuracies else 0
    best_acc = max(metrics.train_accuracies) if metrics.train_accuracies else 0
    initial_loss = metrics.train_losses[0] if metrics.train_losses else 0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    
    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Dataset Statistics', ''],
        ['Total Images', f'{num_images:,}'],
        ['Number of Identities', f'{num_classes:,}'],
        ['Image Size', f'{IMG_SIZE}x{IMG_SIZE}'],
        ['', ''],
        ['Training Configuration', ''],
        ['Model Architecture', 'ResNet-50'],
        ['Embedding Size', f'{EMBEDDING_SIZE}'],
        ['Loss Function', 'ArcFace'],
        ['Batch Size', f'{BATCH_SIZE}'],
        ['Total Epochs', f'{EPOCHS}'],
        ['Initial Learning Rate', f'{LEARNING_RATE}'],
        ['', ''],
        ['Training Results', ''],
        ['Final Loss', f'{final_loss:.4f}'],
        ['Final Accuracy', f'{final_acc:.2f}%'],
        ['Best Accuracy', f'{best_acc:.2f}%'],
        ['Loss Reduction', f'{loss_reduction:.1f}%'],
        ['', ''],
        ['Training Time', ''],
        ['Total Time', f'{total_time/60:.1f} minutes'],
        ['Avg Time/Epoch', f'{avg_time:.1f} seconds'],
        ['Device Used', str(DEVICE).upper()],
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i, row in enumerate(table_data):
        cell = table[(i, 0)]
        if i == 0:  # Header
            cell.set_facecolor('#3498db')
            cell.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif row[0] in ['Dataset Statistics', 'Training Configuration', 'Training Results', 'Training Time']:
            cell.set_facecolor('#95a5a6')
            cell.set_text_props(weight='bold')
        elif row[0] == '':
            cell.set_facecolor('#ecf0f1')
            table[(i, 1)].set_facecolor('#ecf0f1')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
                table[(i, 1)].set_facecolor('#f8f9fa')
    
    plt.title('Training Summary Report', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Summary table saved to {save_path}")
    plt.close()


def plot_batch_progress(metrics, save_path):
    """Plot batch-level progress for first epoch"""
    if len(metrics.batch_losses) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    batches = list(range(1, len(metrics.batch_losses) + 1))
    
    # Batch loss
    ax1.plot(batches, metrics.batch_losses, alpha=0.3, color='#e74c3c', linewidth=0.5)
    # Add moving average
    window = 50
    if len(metrics.batch_losses) > window:
        moving_avg = np.convolve(metrics.batch_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(batches)+1), moving_avg, color='#c0392b', 
                linewidth=2, label=f'{window}-batch moving average')
        ax1.legend()
    
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Batch-Level Loss Progress', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Batch accuracy
    ax2.plot(batches, metrics.batch_accuracies, alpha=0.3, color='#2ecc71', linewidth=0.5)
    # Add moving average
    if len(metrics.batch_accuracies) > window:
        moving_avg = np.convolve(metrics.batch_accuracies, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(batches)+1), moving_avg, color='#27ae60',
                linewidth=2, label=f'{window}-batch moving average')
        ax2.legend()
    
    ax2.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Batch-Level Accuracy Progress', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Batch progress plot saved to {save_path}")
    plt.close()


def save_checkpoint(model, criterion, optimizer, epoch, loss, acc):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'num_classes': criterion.num_classes,
        'embedding_size': EMBEDDING_SIZE
    }
    
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved: {path}")
    
    # Also save as latest
    latest_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)


# ================= MAIN =================
def main():
    import time
    
    print("\n" + "="*60)
    print("FACE RECOGNITION TRAINING - VGGFace2")
    print("="*60 + "\n")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Load dataset
    print("📂 Loading dataset...")
    train_dataset = VGGFace2Dataset(DATA_ROOT, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    num_classes = train_dataset.num_classes
    num_images = len(train_dataset)
    
    # Initialize model
    print("\n🔧 Building model...")
    model = ResNetFace(num_layers=50, embedding_size=EMBEDDING_SIZE).to(DEVICE)
    criterion = ArcFaceLoss(EMBEDDING_SIZE, num_classes, MARGIN, SCALE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        [{'params': model.parameters()}, {'params': criterion.parameters()}],
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 8],
        gamma=0.1
    )
    
    print(f"\n🚀 Starting training...")
    print(f"Training on {num_images:,} images from {num_classes:,} identities\n")
    
    training_start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        loss, acc = train_epoch(model, criterion, train_loader, optimizer, epoch, metrics_tracker)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track epoch metrics
        metrics_tracker.add_epoch(loss, acc, current_lr, epoch_time)
        
        scheduler.step()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS} Summary:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        if epoch % SAVE_INTERVAL == 0:
            save_checkpoint(model, criterion, optimizer, epoch, loss, acc)
    
    total_training_time = time.time() - training_start_time
    
    # Final checkpoint
    print("\n💾 Saving final model...")
    save_checkpoint(model, criterion, optimizer, EPOCHS, loss, acc)
    
    # Save model for inference
    inference_model = {
        'model_state_dict': model.state_dict(),
        'embedding_size': EMBEDDING_SIZE,
        'num_classes': num_classes,
        'img_size': IMG_SIZE
    }
    torch.save(inference_model, os.path.join(CHECKPOINT_DIR, 'face_recognition_model.pth'))
    
    print("\n📊 Generating visualizations and reports...")
    
    # Save metrics to JSON
    metrics_tracker.save_json(os.path.join(OUTPUT_DIR, 'training_metrics.json'))
    
    # Create all visualizations
    plot_training_curves(metrics_tracker, os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plot_loss_accuracy_combined(metrics_tracker, os.path.join(OUTPUT_DIR, 'loss_accuracy_combined.png'))
    create_summary_table(metrics_tracker, num_classes, num_images, os.path.join(OUTPUT_DIR, 'training_summary.png'))
    plot_batch_progress(metrics_tracker, os.path.join(OUTPUT_DIR, 'batch_progress.png'))
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📈 Final Results:")
    print(f"  Final Loss: {loss:.4f}")
    print(f"  Final Accuracy: {acc:.2f}%")
    print(f"  Best Accuracy: {max(metrics_tracker.train_accuracies):.2f}%")
    print(f"  Total Training Time: {total_training_time/60:.1f} minutes")
    print(f"\n📁 Output files saved in:")
    print(f"  Models: {CHECKPOINT_DIR}/")
    print(f"  Visualizations: {OUTPUT_DIR}/")
    print(f"\n✅ All outputs ready for your presentation!\n")


if __name__ == "__main__":
    main()
