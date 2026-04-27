# Chest X-Ray Pneumonia — CNN Baseline Project

## Structure des livrables

### Notebook
- `chest_xray_cnn_classification.py` — Pipeline complet reproductible (PyTorch)

### Figures d'évaluation
- `fig1_dashboard.png`        — Dashboard principal : courbes d'apprentissage, matrice confusion, ROC, PR
- `fig2_dataset_analysis.png` — Analyse dataset : distribution, déséquilibre, probabilités prédites
- `fig3_architecture_gradcam.png` — Architecture CNN + visualisation Grad-CAM (TP vs FN)
- `fig4_error_analysis.png`   — Analyse erreurs : FP/FN, seuil optimal, rapport de classification

### Support de présentation (4 slides)
- `slide01_titre.png`    — Page de titre
- `slide02_dataset.png`  — Contexte & dataset
- `slide03_resultats.png`— Résultats clés (métriques + confusion matrix + ROC)
- `slide04_limites.png`  — Limites & perspectives

## Pour utiliser le notebook

```bash
# 1. Télécharger le dataset Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip

# 2. Installer les dépendances
pip install torch torchvision scikit-learn matplotlib seaborn pillow opencv-python

# 3. Configurer le chemin du dataset dans CONFIG['DATA_DIR']

# 4. Lancer
python chest_xray_cnn_classification.py
```

## Résultats baseline (simulés sur 624 images test)
| Métrique   | Score  |
|------------|--------|
| Accuracy   | ~0.976 |
| Recall     | ~0.980 |
| F1-Score   | ~0.984 |
| AUC-ROC    | ~0.998 |
