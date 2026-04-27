
# 0. IMPORTS & CONFIGURATION

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Deep Learning — PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Métriques & évaluation
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, recall_score, precision_score, accuracy_score
)

# Reproductibilité 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

#  Device 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#  Hyperparamètres 
CONFIG = {
    # Données
    'DATA_DIR':      r'C:\Users\AROL TT COURT\ProjetB\chest-xray-pneumonia\chest_xray',
    'IMG_SIZE':      224,
    'BATCH_SIZE':    32,
    'NUM_WORKERS':   4,
    # Entraînement
    'EPOCHS':        25,
    'LR':            1e-3,
    'WEIGHT_DECAY':  1e-4,
    'LR_PATIENCE':   5,
    'LR_FACTOR':     0.5,
    # Modèle
    'DROPOUT':       0.5,
    'NUM_CLASSES':   1,                   # Binaire → Sigmoid
    # Chemins sortie
    'OUTPUT_DIR':    './outputs',
    'MODEL_PATH':    './outputs/best_model.pth',
}

Path(CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
print(f"\n Config:")
for k, v in CONFIG.items():
    print(f"   {k:<18} = {v}")


# 1. PIPELINE DE DONNÉES

#  Transformations 
# Normalisation ImageNet (standard pour transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.Grayscale(num_output_channels=3),    # X-rays → 3ch pour CNN RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

#  Chargement 
def load_datasets(data_dir):
    """
    Structure attendue du dossier (identique à Kaggle) :
    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
    """
    data_dir = Path(data_dir)
    
    train_ds = datasets.ImageFolder(data_dir / 'train', transform=train_transforms)
    val_ds   = datasets.ImageFolder(data_dir / 'val',   transform=val_test_transforms)
    test_ds  = datasets.ImageFolder(data_dir / 'test',  transform=val_test_transforms)
    
    print(f"\n Dataset chargé:")
    print(f"   Classes : {train_ds.classes}")
    print(f"   Train   : {len(train_ds):,} images")
    print(f"   Val     : {len(val_ds):,}   images")
    print(f"   Test    : {len(test_ds):,}  images")
    
    # Compter classes
    for split_name, ds in [('Train', train_ds), ('Test', test_ds)]:
        counts = np.bincount([label for _, label in ds])
        for cls_idx, cls_name in enumerate(ds.classes):
            pct = 100 * counts[cls_idx] / len(ds)
            print(f"   {split_name} {cls_name}: {counts[cls_idx]:,} ({pct:.1f}%)")
    
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, config):
    """
    Gestion du déséquilibre de classes avec WeightedRandomSampler.
    PNEUMONIA (74%) >> NORMAL (26%) → Surech. de NORMAL en train.
    """
    # Poids inversement proportionnels à la fréquence de classe
    class_counts  = np.bincount([label for _, label in train_ds])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in train_ds]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=config['BATCH_SIZE'],
        sampler=sampler, num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['BATCH_SIZE'],
        shuffle=False, num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['BATCH_SIZE'],
        shuffle=False, num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )
    
    print(f"\n   WeightedRandomSampler activé (rééquilibrage train)")
    print(f"   Batches train: {len(train_loader)}")
    return train_loader, val_loader, test_loader


#  Visualisation d'exemples
def show_sample_images(dataset, n=8, save_path=None):
    """Affiche n images du dataset avec leurs labels."""
    fig, axes = plt.subplots(2, n//2, figsize=(14, 6))
    fig.suptitle('Exemples du dataset Chest X-Ray', fontsize=13, fontweight='bold')
    
    indices = random.sample(range(len(dataset)), n)
    for idx, (ax, sample_idx) in enumerate(zip(axes.flatten(), indices)):
        img_tensor, label = dataset[sample_idx]
        # Dénormaliser
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        # Grayscale car radiographie
        ax.imshow(img[:, :, 0], cmap='bone')
        cls_name = dataset.classes[label]
        color = '#f85149' if label == 1 else '#58a6ff'
        ax.set_title(cls_name, color=color, fontsize=9, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# 2. ARCHITECTURE CNN BASELINE (FROM SCRATCH)

class ChestXRayCNN(nn.Module):
    """
    CNN baseline pour classification binaire NORMAL/PNEUMONIA.
    
    Architecture :
    ┌─────────────────────────────────────────────────────────┐
    │  Input: [B, 3, 224, 224]                                │
    │                                                         │
    │  Block 1: Conv2d(3→32, 3×3) + BN + ReLU + MaxPool      │
    │  Block 2: Conv2d(32→64, 3×3) + BN + ReLU + MaxPool     │
    │  Block 3: Conv2d(64→128, 3×3) + BN + ReLU + MaxPool    │
    │  Block 4: Conv2d(128→256, 3×3) + BN + ReLU + MaxPool   │
    │                                                         │
    │  AdaptiveAvgPool(4×4) → Flatten                         │
    │  FC(4096 → 512) + ReLU + Dropout(0.5)                  │
    │  FC(512 → 1) + Sigmoid                                  │
    │                                                         │
    │  Paramètres : ~8.5M                                     │
    └─────────────────────────────────────────────────────────┘
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        
        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            conv_block(3,   32),     # 224 → 112
            conv_block(32,  64),     # 112 → 56
            conv_block(64,  128),    # 56  → 28
            conv_block(128, 256),    # 28  → 14
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        
        # Initialisation des poids (He)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x   


#  Variante Transfer Learning (ResNet18) 
class ChestXRayResNet(nn.Module):
    """
    Alternative avec ResNet18 pré-entraîné sur ImageNet.
    Fine-tuning : on gèle toutes les couches sauf les 2 derniers blocs + head.
    Meilleure performance, mais plus de paramètres (~11M).
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Geler les premières couches
        for name, param in backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        n_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1),
        )
        self.model = backbone
    
    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Paramètres du modèle:")
    print(f"   Total       : {total:,}")
    print(f"   Entraînables: {trainable:,}")
    return total, trainable


# 3. ENTRAÎNEMENT

class Trainer:
    """
    Classe d'entraînement avec :
    - Checkpointing du meilleur modèle (val AUC)
    - Early stopping
    - Learning rate scheduling
    - Logging des métriques
    """
    def __init__(self, model, config, class_weights=None):
        self.model  = model.to(DEVICE)
        self.config = config
        
        # Loss avec pondération de classe (alternative au Sampler)
        pos_weight = torch.tensor([class_weights]) if class_weights else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['LR'],
            weight_decay=config['WEIGHT_DECAY']
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='max',
            patience=config['LR_PATIENCE'],
            factor=config['LR_FACTOR']
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  [],
            'val_auc':    [], 'val_f1':   [],
            'lr':         []
        }
        self.best_val_auc  = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.EARLY_STOP_PATIENCE = 8
    
    def _run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        all_labels = []
        all_probs  = []
        
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in loader:
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
                
                if train:
                    self.optimizer.zero_grad()
                
                logits = self.model(imgs)
                loss   = self.criterion(logits, labels)
                
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).squeeze().cpu().detach().numpy()
                all_probs.extend(probs.flatten().tolist())
                all_labels.extend(labels.squeeze().cpu().numpy().tolist())
        
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)
        all_preds  = (all_probs > 0.5).astype(int)
        
        avg_loss = total_loss / len(loader.dataset)
        acc  = accuracy_score(all_labels, all_preds)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        
        if not train:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc     = auc(fpr, tpr)
        else:
            roc_auc = None
        
        return avg_loss, acc, f1, roc_auc, all_labels, all_probs
    
    def train(self, train_loader, val_loader):
        print(f"\n{'='*60}")
        print(f"  DÉMARRAGE ENTRAÎNEMENT — {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  "
              f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValAUC':>7}  {'LR':>8}")
        print(f"  {'─'*58}")
        
        for epoch in range(1, self.config['EPOCHS'] + 1):
            # ── Entraînement ──
            tr_loss, tr_acc, tr_f1, _, _, _ = self._run_epoch(train_loader, train=True)
            
            # ── Validation ──
            val_loss, val_acc, val_f1, val_auc, _, _ = self._run_epoch(val_loader, train=False)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_auc)
            
            # ── Logging ──
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            marker = ''
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'config': self.config,
                }, self.config['MODEL_PATH'])
                marker = '   saved'
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            print(f"  {epoch:>4}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
                  f"{val_loss:>8.4f}  {val_acc*100:>6.2f}%  {val_auc:>6.4f}  "
                  f"{current_lr:>8.2e}{marker}")
            
            # ── Early stopping ──
            if self.patience_counter >= self.EARLY_STOP_PATIENCE:
                print(f"\n    Early stopping (patience={self.EARLY_STOP_PATIENCE})")
                break
        
        print(f"\n   Entraînement terminé. Meilleur AUC val: {self.best_val_auc:.4f}")
        return self.history


# 4. ÉVALUATION COMPLÈTE

def evaluate_model(model, test_loader, checkpoint_path=None):
    """
    Évaluation sur le test set avec toutes les métriques.
    Charge le meilleur checkpoint si disponible.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n Checkpoint chargé (epoch {checkpoint['epoch']}, "
              f"val_auc={checkpoint['val_auc']:.4f})")
    
    model.eval()
    all_labels = []
    all_probs  = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.numpy().tolist())
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    
    #  Métriques 
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)  
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    pr_auc  = auc(rec_c, prec_c)
    cm      = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  RÉSULTATS — TEST SET  (n={len(y_true)})")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Recall      : {rec:.4f}  (Sensibilité)")
    print(f"  Spécificité : {spec:.4f}")
    print(f"  Précision   : {prec:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {roc_auc:.4f}")
    print(f"  AUC-PR      : {pr_auc:.4f}")
    print(f"\n  Matrice de Confusion:")
    print(f"  {'':12} Prédit NORMAL  Prédit PNEUMONIA")
    print(f"  Réel NORMAL   {cm[0,0]:>12}  {cm[0,1]:>16}")
    print(f"  Réel PNEUMONIA{cm[1,0]:>12}  {cm[1,1]:>16}")
    print(f"\n  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\n   Faux Négatifs (pneumonies manquées): {cm[1,0]}")
    print(f"     Coût clinique maximal → minimiser FN")
    
    print(f"\n{classification_report(y_true, y_pred, target_names=['NORMAL','PNEUMONIA'])}")
    
    return {
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': acc, 'rec': rec, 'prec': prec, 'f1': f1,
        'spec': spec, 'auc_roc': roc_auc, 'auc_pr': pr_auc,
        'fpr': fpr, 'tpr': tpr, 'cm': cm,
    }


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Trouve le seuil optimal selon la métrique choisie.
    En clinique : souvent 'recall' pour maximiser la sensibilité.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_score = 0
    best_t     = 0.5
    
    for t in thresholds:
        y_p = (y_prob > t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_p, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_p, zero_division=0)
        elif metric == 'balanced':
            rec  = recall_score(y_true, y_p, zero_division=0)
            spec = recall_score(y_true, y_p, pos_label=0, zero_division=0)
            score = (rec + spec) / 2  # Balanced accuracy
        
        if score > best_score:
            best_score = score
            best_t = t
    
    print(f"\n Seuil optimal ({metric}): {best_t:.3f}  (score: {best_score:.4f})")
    return best_t, best_score


# 5. VISUALISATIONS

def plot_training_history(history, save_path=None):
    """Courbes d'apprentissage : Loss et Accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', ms=4, label='Train')
    ax1.plot(epochs, history['val_loss'],   'r-o', ms=4, label='Val')
    ax1.set_title('Loss (BCE)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Époque'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'b-o', ms=4, label='Train')
    ax2.plot(epochs, [a*100 for a in history['val_acc']],   'r-o', ms=4, label='Val')
    ax2.plot(epochs, [a*100 for a in history['val_auc']],   'g--', lw=2, label='Val AUC')
    ax2.set_title('Accuracy & AUC', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Époque'); ax2.set_ylabel('Score (%)')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(50, 101)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_evaluation_dashboard(results, save_path=None):
    """Dashboard complet d'évaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Évaluation du Modèle CNN — Chest X-Ray',
                 fontsize=14, fontweight='bold')
    
    # ── 1. Matrice de confusion ──────────────────────────────────
    ax = axes[0, 0]
    cm = results['cm']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=False, fmt='.2%', cmap='Blues',
                xticklabels=['NORMAL','PNEUMONIA'],
                yticklabels=['NORMAL','PNEUMONIA'], ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.5,
                    f'{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if cm_norm[i,j] > 0.5 else 'black')
    ax.set_title('Matrice de Confusion', fontsize=12)
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    
    # ── 2. Courbe ROC ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(results['fpr'], results['tpr'], 'b-', lw=2,
            label=f"AUC = {results['auc_roc']:.4f}")
    ax.fill_between(results['fpr'], results['tpr'], alpha=0.1)
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_title('Courbe ROC', fontsize=12)
    ax.set_xlabel('FPR (1-Spécificité)')
    ax.set_ylabel('TPR (Sensibilité / Recall)')
    ax.legend(); ax.grid(alpha=0.3)
    
    #  3. Distribution des probabilités 
    ax = axes[1, 0]
    prob_n = results['y_prob'][results['y_true'] == 0]
    prob_p = results['y_prob'][results['y_true'] == 1]
    ax.hist(prob_n, bins=30, alpha=0.7, label='NORMAL',    color='steelblue', density=True)
    ax.hist(prob_p, bins=30, alpha=0.7, label='PNEUMONIA', color='firebrick', density=True)
    ax.axvline(0.5, color='k', ls='--', lw=1.5, label='Seuil 0.5')
    ax.set_title('Distribution des Scores de Confiance', fontsize=12)
    ax.set_xlabel('P(PNEUMONIA)'); ax.set_ylabel('Densité')
    ax.legend(); ax.grid(alpha=0.3)
    
    #  4. Résumé métriques 
    ax = axes[1, 1]
    ax.axis('off')
    metrics = {
        'Accuracy':    results['acc'],
        'Recall':      results['rec'],
        'Spécificité': results['spec'],
        'Précision':   results['prec'],
        'F1-Score':    results['f1'],
        'AUC-ROC':     results['auc_roc'],
        'AUC-PR':      results['auc_pr'],
    }
    y = 0.95
    ax.text(0.5, y, 'Métriques finales — Test Set', ha='center',
            fontsize=13, fontweight='bold', transform=ax.transAxes)
    for name, val in metrics.items():
        y -= 0.11
        color = '#2ca02c' if val >= 0.90 else '#d62728' if val < 0.80 else '#ff7f0e'
        ax.text(0.15, y, name, fontsize=11, transform=ax.transAxes)
        ax.text(0.75, y, f'{val:.4f}', fontsize=11, fontweight='bold',
                color=color, transform=ax.transAxes)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# 6. GRAD-CAM

class GradCAM:
    """
    Gradient-weighted Class Activation Maps (Selvaraju et al., 2017).
    
    Principe :
    1. Forward pass → sortie de la dernière couche conv + logit final
    2. Backward pass → gradients par rapport aux feature maps
    3. Global Average Pooling des gradients → poids par canal
    4. Combinaison linéaire pondérée des feature maps → heatmap
    5. ReLU + upsampling → superposition sur l'image
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, img_tensor):
        """
        Args:
            img_tensor : (1, 3, H, W) — image normalisée
        Returns:
            heatmap     : (H, W) — carte d'activation [0, 1]
            pred_prob   : float — probabilité PNEUMONIA
        """
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True
        
        # Forward
        logit = self.model(img_tensor)
        pred_prob = torch.sigmoid(logit).item()
        
        # Backward
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))
        
        # Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP des gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalisation + resize
        import cv2
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, pred_prob


def visualize_gradcam(model, dataset, indices, save_path=None):
    """
    Visualise les Grad-CAM pour une liste d'indices du dataset.
    À adapter selon l'architecture (target_layer = dernière conv).
    """
    # Récupérer la dernière couche conv
    target_layer = list(model.features.children())[-1][-3]  # Dernière Conv2d
    gcam = GradCAM(model, target_layer)
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for row, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        
        # Image originale (dénormalisée)
        img_np = img_tensor.permute(1,2,0).numpy()
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)[:, :, 0]  # Channel 0 (grayscale)
        
        # Grad-CAM
             
        heatmap, pred_prob = gcam.generate(img_tensor)
        
        # Overlay
        import cv2
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
        img_rgb = np.stack([img_np]*3, axis=-1)
        overlay = 0.6 * img_rgb + 0.4 * heatmap_color
        overlay = np.clip(overlay, 0, 1)
        
        cls_name  = dataset.classes[label]
        pred_cls  = 'PNEUMONIA' if pred_prob > 0.5 else 'NORMAL'
        correct   = '✓' if cls_name == pred_cls else '✗'
        
        axes[row][0].imshow(img_np, cmap='bone')
        axes[row][0].set_title(f'Original\nLabel: {cls_name}', fontsize=9)
        axes[row][0].axis('off')
        
        axes[row][1].imshow(heatmap, cmap='jet')
        axes[row][1].set_title(f'Heatmap Grad-CAM\nZones activées', fontsize=9)
        axes[row][1].axis('off')
        
        color = '#2ca02c' if correct == '✓' else '#d62728'
        axes[row][2].imshow(overlay)
        axes[row][2].set_title(
            f'Overlay {correct}\nPrédit: {pred_cls} ({pred_prob:.3f})',
            fontsize=9, color=color)
        axes[row][2].axis('off')
    
    plt.suptitle('Grad-CAM — Interprétabilité du CNN', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# 7. PIPELINE PRINCIPAL (MAIN)

def main():
    print("CHEST X-RAY PNEUMONIA — PIPELINE CNN ")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed : {SEED}\n")
    
    #  Étape 1 : Données
    print("【1/6】 Chargement des données...")
    train_ds, val_ds, test_ds = load_datasets(CONFIG['DATA_DIR'])
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, CONFIG)
    
    # Visualiser quelques exemples
    show_sample_images(train_ds, n=8,
                       save_path=f"{CONFIG['OUTPUT_DIR']}/sample_images.png")
    
    #  Étape 2 : Modèle 
    print("\n【2/6】 Construction du modèle...")
    model = ChestXRayCNN(dropout=CONFIG['DROPOUT'])
    count_parameters(model)
    
    #  Étape 3 : Entraînement 
    print("\n【3/6】 Entraînement...")
    # Calcul du poids positif (ratio NORMAL/PNEUMONIA)
    n_normal    = sum(1 for _, l in train_ds if l == 0)
    n_pneumonia = sum(1 for _, l in train_ds if l == 1)
    pos_weight  = n_normal / n_pneumonia
    print(f"   Pos weight (BCE): {pos_weight:.3f}")
    
    trainer = Trainer(model, CONFIG, class_weights=pos_weight)
    history = trainer.train(train_loader, val_loader)
    
    # Courbes d'apprentissage
    plot_training_history(
        history,
        save_path=f"{CONFIG['OUTPUT_DIR']}/training_curves.png")
    
    #  Étape 4 : Évaluation
    print("\n【4/6】 Évaluation sur le test set...")
    results = evaluate_model(model, test_loader, CONFIG['MODEL_PATH'])
    
    # Optimisation du seuil
    t_f1, _   = find_optimal_threshold(results['y_true'], results['y_prob'], 'f1')
    t_rec, _  = find_optimal_threshold(results['y_true'], results['y_prob'], 'recall')
    t_bal, _  = find_optimal_threshold(results['y_true'], results['y_prob'], 'balanced')
    
    print(f"\n  Résultats avec seuil optimal (F1={t_f1:.3f}):")
    y_pred_opt = (results['y_prob'] > t_f1).astype(int)
    print(f"  F1={f1_score(results['y_true'], y_pred_opt):.4f}  "
          f"Recall={recall_score(results['y_true'], y_pred_opt):.4f}")
    
    # Dashboard d'évaluation
    plot_evaluation_dashboard(
        results,
        save_path=f"{CONFIG['OUTPUT_DIR']}/evaluation_dashboard.png")
    
    #  Étape 5 : Grad-CAM 
    print("\n【5/6】 Visualisation Grad-CAM...")
    # Sélectionner des cas intéressants (TP, TN, FP, FN)
    y_t = results['y_true']
    y_p = results['y_pred']
    tp_idx = np.where((y_t == 1) & (y_p == 1))[0][:2]
    fn_idx = np.where((y_t == 1) & (y_p == 0))[0][:2]
    
    model_for_cam = ChestXRayCNN(dropout=CONFIG['DROPOUT'])
    ck = torch.load(CONFIG['MODEL_PATH'], map_location=DEVICE)
    model_for_cam.load_state_dict(ck['model_state_dict'])
    
    visualize_gradcam(
        model_for_cam, test_ds,
        indices=list(tp_idx) + list(fn_idx),
        save_path=f"{CONFIG['OUTPUT_DIR']}/gradcam.png")
    
    #  Étape 6 : Résumé final
    print("\n【6/6】 Résumé final")
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           RÉSULTATS FINAUX — TEST SET                     ║
╠═══════════════════════════════════════════════════════════╣
║  Accuracy    : {results['acc']:.4f}                                ║
║  Recall      : {results['rec']:.4f}  (Sensibilité)                ║
║  Spécificité : {results['spec']:.4f}                               ║
║  Précision   : {results['prec']:.4f}                               ║
║  F1-Score    : {results['f1']:.4f}                                 ║
║  AUC-ROC     : {results['auc_roc']:.4f}                            ║
║  AUC-PR      : {results['auc_pr']:.4f}                             ║
╠═══════════════════════════════════════════════════════════╣
║  FN (manqués): {results['cm'][1,0]:>4}  (pneumonies non détectées) ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    return model, history, results


# ── Point d'entrée ─────────────────────────────────────────────
if __name__ == '__main__':
    model, history, results = main()
