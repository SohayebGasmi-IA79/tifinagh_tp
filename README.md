<div align="center">

# 🔤 Classification des Caractères Tifinagh
### avec un Réseau de Neurones Multiclasses (MLP) — *from scratch*

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/License-Academic-green?style=for-the-badge" alt="License">

<br>

**SOHAYEB GASMI** — Masters **IAA** 

<em>Implémentation complète d'un réseau de neurones multicouches sans framework de Deep Learning : forward, backprop, Adam, L2, K-Fold et augmentation de données.</em>

<br>

<a href="https://github.com/SohayebGasmi-IA79/tifinagh_tp">
  <img src="https://img.shields.io/badge/Repo-tifinagh__tp-181717?style=for-the-badge&logo=github&logoColor=white" alt="Repo">
</a>

</div>

---

## 📑 Table des matières

- [✨ Aperçu](#-aperçu)
- [🚀 Fonctionnalités](#-fonctionnalités)
- [🧠 Architecture du réseau](#-architecture-du-réseau)
- [📂 Structure du projet](#-structure-du-projet)
- [⚙️ Installation](#️-installation)
- [🗂️ Dataset](#️-dataset)
- [▶️ Utilisation](#️-utilisation)
- [📊 Résultats & Visualisations](#-résultats--visualisations)
- [🔬 Détails mathématiques](#-détails-mathématiques)
- [🛠️ Personnalisation](#️-personnalisation)


---

## ✨ Aperçu

Ce projet implémente **entièrement à la main** (sans PyTorch / TensorFlow / Keras) un **Perceptron Multicouche (MLP)** capable de classifier **33 caractères Tifinagh** manuscrits provenant du dataset **AMHCD** (*Amazigh Handwritten Character Database*).

<div align="center">

| Composant | Implémentation |
|:---:|:---:|
| Forward propagation | `NumPy` pur |
| Backpropagation | Dérivées analytiques |
| Activation cachée | **ReLU** |
| Activation sortie | **Softmax** |
| Perte | **Categorical Cross-Entropy** + **L2** |
| Optimiseur | **SGD** ou **Adam** |

</div>

---

## 🚀 Fonctionnalités

<table>
<tr>
<td width="50%" valign="top">

### 🎯 Cœur du modèle
- ✅ Réseau multicouche configurable (`layer_sizes`)
- ✅ Initialisation **He** (`sqrt(2/fan_in)`)
- ✅ Stabilité numérique du Softmax
- ✅ Assertions partout pour valider les shapes

</td>
<td width="50%" valign="top">

### 🎁 Bonus implémentés
- ✅ **Régularisation L2** (`lambda_reg`)
- ✅ **Optimiseur Adam** (`β₁=0.9`, `β₂=0.999`, `ε=1e-8`)
- ✅ **Validation croisée K-Fold**
- ✅ **Augmentation de données** (rotations + translations)
- ✅ Mini-batch training avec shuffle

</td>
</tr>
</table>

---

## 🧠 Architecture du réseau

<div align="center">

```
   Entrée (1024)      Cachée 1 (64)      Cachée 2 (32)      Sortie (33)
   ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
   │  32x32   │  ───▶ │   ReLU   │  ───▶ │   ReLU   │  ───▶ │ Softmax  │
   │  image   │       │          │       │          │       │          │
   └──────────┘       └──────────┘       └──────────┘       └──────────┘
       1024               64                 32                 33
```

</div>

| Couche | Taille | Activation |
|:---:|:---:|:---:|
| Entrée | `32 × 32 = 1024` | — |
| Cachée 1 | `64` | ReLU |
| Cachée 2 | `32` | ReLU |
| Sortie | `33` (classes) | Softmax |

---

## 📂 Structure du projet

```bash
📁 tifinagh_tp/
├── 📄 tp.py                                   # Script principal (tout-en-un)
├── 📓 tp_tifinagh.ipynb                       # Version notebook Jupyter
├── 📄 README.md                               # Ce fichier
│
├── 📁 amazigh-handwritten-character-database-amhcd/
│   ├── 📁 amhcd_64/
│   │   └── 📁 AMHCD_64/                       # ✅ Dossier réellement utilisé
│   │       ├── 📁 ya/                         # Classe 1
│   │       ├── 📁 yab/                        # Classe 2
│   │       ├── 📁 yach/                       # Classe 3
│   │       ├── 📁 yad/
│   │       ├── 📁 yadd/
│   │       ├── 📁 yae/
│   │       ├── 📁 yaf/
│   │       ├── 📁 yag/
│   │       ├── 📁 yagh/
│   │       ├── 📁 yagw/
│   │       ├── 📁 yah/
│   │       ├── 📁 yahh/
│   │       ├── 📁 yaj/
│   │       ├── 📁 yak/
│   │       ├── 📁 yakw/
│   │       ├── 📁 yal/
│   │       ├── 📁 yam/
│   │       ├── 📁 yan/
│   │       ├── 📁 yaq/
│   │       ├── 📁 yar/
│   │       ├── 📁 yarr/
│   │       ├── 📁 yas/
│   │       ├── 📁 yass/
│   │       ├── 📁 yat/
│   │       ├── 📁 yatt/
│   │       ├── 📁 yaw/
│   │       ├── 📁 yax/
│   │       ├── 📁 yay/
│   │       ├── 📁 yaz/
│   │       ├── 📁 yazz/
│   │       ├── 📁 yey/
│   │       ├── 📁 yi/
│   │       └── 📁 yu/                         # Classe 33
│   │
│   ├── 📁 AMHCD_64/                           
│   │   └── 📁 AMHCD_64/
│   │       ├── 📁 ya/ … 📁 yu/                # (33 dossiers identiques)
│   │
│   └── 📁 labels/
│       └── 📁 labels/
│           ├── 📄 33-common-latin-tifinagh.txt
│           └── 📄 sorted-33-common-tifinagh.txt
│
├── 🖼️ confusion_matrix.png                    # Généré à l'exécution
└── 📈 loss_accuracy_plot.png                  # Généré à l'exécution
```

> 💡 **Note** : le script utilise le chemin `amazigh-handwritten-character-database-amhcd/amhcd_64/AMHCD_64/` — c'est la copie contenant les 33 classes utilisées pour l'entraînement.

---

## ⚙️ Installation

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/SohayebGasmi-IA79/tifinagh_tp.git
cd tifinagh_tp
```

### 2️⃣ Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3️⃣ Installer les dépendances
```bash
pip install numpy pandas opencv-python scikit-learn matplotlib seaborn
```

---

## 🗂️ Dataset

Le projet utilise le dataset **AMHCD (Amazigh Handwritten Character Database)** :

<div align="center">

| Caractéristique | Valeur |
|:---|:---:|
| Nombre de classes | **33** |
| Taille des images | **64 × 64** (redimensionnées à **32 × 32**) |
| Format | PNG (niveaux de gris) |
| Organisation | 1 dossier = 1 classe |
| Nom des classes | `ya`, `yab`, `yach`, …, `yu` |

</div>

### 📖 Correspondance des classes

Les 33 classes sont nommées selon **la translittération latine** du caractère Tifinagh correspondant. Le fichier de référence se trouve ici :

```
amazigh-handwritten-character-database-amhcd/labels/labels/sorted-33-common-tifinagh.txt
```

> ⚠️ **Important** : Dans cette version, il n'y a **pas de fichier `labels-map.csv`**. Le label est directement déduit du **nom du dossier parent** de chaque image.

---

## ▶️ Utilisation

### 🏃 Exécution du script principal

```bash
python tp.py
```

### 📓 Ou utilisation du notebook

```bash
jupyter notebook tp_tifinagh.ipynb
```

### 🎛️ Options configurables dans le script

```python
# --- Augmentation de données ---
USE_DATA_AUGMENTATION = True       # activer/désactiver

# --- Validation croisée ---
RUN_KFOLD = True                   # activer/désactiver

# --- Hyperparamètres du modèle final ---
layer_sizes   = [1024, 64, 32, 33]
learning_rate = 0.001
optimizer     = "adam"             # ou "sgd"
lambda_reg    = 0.0001
epochs        = 150
batch_size    = 32
```

### 📤 Sorties générées

<div align="center">

| Fichier | Description |
|:---:|:---:|
| `confusion_matrix.png` | Matrice de confusion sur le test set |
| `loss_accuracy_plot.png` | Courbes de perte & précision (train/val) |

</div>

---

## 📊 Résultats & Visualisations

### 📈 Courbes d'apprentissage

Les courbes générées permettent de diagnostiquer :

- **Underfitting** : perte train élevée et stagnante
- **Overfitting** : écart croissant train / val
- **Convergence** : stabilisation de la val loss

### 🔥 Matrice de confusion

Affiche pour chaque caractère Tifinagh (33×33) :
- ✅ Les bonnes classifications (diagonale)
- ❌ Les confusions entre glyphes similaires (ex. `yad` vs `yadd`, `yat` vs `yatt`)

### 📋 Rapport de classification

Le script affiche un `classification_report` complet :

```
              precision    recall  f1-score   support

          ya       0.95      0.93      0.94        30
         yab       0.92      0.96      0.94        28
        yach       0.94      0.91      0.92        29
        ...        ...       ...       ...       ...

    accuracy                           0.93       990
   macro avg       0.93      0.92      0.92       990
weighted avg       0.93      0.93      0.93       990
```

---

## 🔬 Détails mathématiques

<details>
<summary><b>📐 Forward propagation</b> (cliquez pour déplier)</summary>

<br>

Pour chaque couche `l` :

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}
$$

$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

avec :
- `g = ReLU` pour les couches cachées
- `g = Softmax` pour la couche de sortie

</details>

<details>
<summary><b>📉 Fonction de perte</b> (cliquez pour déplier)</summary>

<br>

**Cross-Entropy catégorielle + pénalité L2** :

$$
J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log(\hat{y}_{ik}) + \frac{\lambda}{2m}\sum_{l}\|W^{[l]}\|_F^2
$$

</details>

<details>
<summary><b>⚡ Optimiseur Adam</b> (cliquez pour déplier)</summary>

<br>

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla W
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla W)^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
W \leftarrow W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

</details>

<details>
<summary><b>🔁 Validation croisée K-Fold</b> (cliquez pour déplier)</summary>

<br>

- Le jeu `train + val` est divisé en **K = 5 folds**
- À chaque itération : `K-1` folds pour l'entraînement, `1` fold pour la validation
- Résultat : moyenne ± écart-type de l'accuracy
- Objectif : **estimer la robustesse** du modèle avant l'entraînement final

</details>

<details>
<summary><b>🔄 Augmentation de données</b> (cliquez pour déplier)</summary>

<br>

Pour chaque image d'entraînement, on génère une version augmentée :

- **Rotation aléatoire** : angle ∈ [-8°, +8°]
- **Translation aléatoire** : décalage (tx, ty) ∈ [-2 px, +2 px]

La transformation est appliquée via `cv2.warpAffine` avec `borderValue = 0.0` (fond noir).

</details>

---

## 🛠️ Personnalisation

<table>
<tr>
<th>Paramètre</th>
<th>Effet</th>
<th>Valeurs conseillées</th>
</tr>
<tr>
<td><code>layer_sizes</code></td>
<td>Profondeur / largeur du réseau</td>
<td><code>[1024, 128, 64, 33]</code></td>
</tr>
<tr>
<td><code>learning_rate</code></td>
<td>Vitesse d'apprentissage</td>
<td><code>0.001</code> (Adam) / <code>0.01</code> (SGD)</td>
</tr>
<tr>
<td><code>lambda_reg</code></td>
<td>Force de la régularisation L2</td>
<td><code>1e-4</code> à <code>1e-2</code></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Taille des mini-batchs</td>
<td><code>32</code> ou <code>64</code></td>
</tr>
<tr>
<td><code>epochs</code></td>
<td>Nombre d'itérations complètes</td>
<td><code>100</code> à <code>300</code></td>
</tr>
<tr>
<td><code>n_augmented_per_sample</code></td>
<td>Nombre d'images augmentées par échantillon</td>
<td><code>1</code> à <code>3</code></td>
</tr>
</table>


</sub>

</div>