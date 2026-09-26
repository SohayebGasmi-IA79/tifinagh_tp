"""
TP - Classification des Caractères Tifinagh avec un Réseau de Neurones Multiclasses (MLP)


Version complète  :
  - Régularisation L2
  - Optimiseur Adam
  - Validation croisée K-fold
  - Augmentation de données (rotations + translations)
"""

import os
import copy
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. Fonctions d'activation
# =========================================================
def relu(x):
    """
    ReLU activation : max(0, x)
    """
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    result = np.maximum(0, x)
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result


def relu_derivative(x):
    """
    Derivative of ReLU : 1 if x > 0, else 0
    """
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    result = (x > 0).astype(float)
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result


def softmax(x):
    """
    Softmax activation : exp(x) / sum(exp(x))
    """
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    # Stabilité numérique : on soustrait le max par ligne avant l'exponentielle
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1), "Softmax output must sum to 1 per sample"
    return result


# =========================================================
# 2. Classe MultiClassNeuralNetwork
# =========================================================
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01,
                 optimizer="sgd", lambda_reg=0.0,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the neural network with given layer sizes and learning rate.
        layer_sizes : List of integers [input_size, hidden1_size, ..., output_size]

        Bonus:
          optimizer : "sgd" ou "adam"
          lambda_reg : coefficient de régularisation L2 (0 = désactivée)
          beta1, beta2, epsilon : hyperparamètres d'Adam
        """
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, \
            "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), \
            "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, \
            "Learning rate must be a positive number"
        assert optimizer in ("sgd", "adam"), "optimizer must be 'sgd' or 'adam'"
        assert lambda_reg >= 0, "lambda_reg must be non-negative"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lambda_reg = lambda_reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weights = []
        self.biases = []

        # Initialisation des poids et biais
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            assert w.shape == (layer_sizes[i], layer_sizes[i + 1]), f"Weight matrix {i + 1} has incorrect shape"
            assert b.shape == (1, layer_sizes[i + 1]), f"Bias vector {i + 1} has incorrect shape"
            self.weights.append(w)
            self.biases.append(b)

        # Etats Adam (moments d'ordre 1 et 2) — initialisés même si non utilisés
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # compteur de pas de temps (pour la correction de biais Adam)

    def forward(self, X):
        """
        Forward propagation : Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, A^{[l]} = g(Z^{[l]})
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], \
            f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

        self.activations = [X]
        self.z_values = []

        # Couches cachées : ReLU
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            assert z.shape == (X.shape[0], self.layer_sizes[i + 1]), f"Z^{[i + 1]} has incorrect shape"
            self.z_values.append(z)
            self.activations.append(relu(z))

        # Couche de sortie : softmax
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        assert z.shape == (X.shape[0], self.layer_sizes[-1]), "Output Z has incorrect shape"
        self.z_values.append(z)
        output = softmax(z)
        assert output.shape == (X.shape[0], self.layer_sizes[-1]), "Output A has incorrect shape"
        self.activations.append(output)

        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        """
        Categorical Cross-Entropy (+ pénalité L2) : J = -1/m * sum(y_true * log(y_pred)) + L2
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / m

        # Bonus : régularisation L2 -> (lambda / (2m)) * sum(W^2)
        l2_penalty = 0.0
        if self.lambda_reg > 0:
            l2_penalty = (self.lambda_reg / (2 * m)) * sum(np.sum(w ** 2) for w in self.weights)

        loss = cross_entropy + l2_penalty
        assert not np.isnan(loss), "Loss computation resulted in NaN"
        return loss

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy : proportion of correct predictions
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy

    def backward(self, X, y, outputs):
        """
        Backpropagation : compute dW^{[l]}, db^{[l]} for each layer
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), \
            "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], \
            f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"

        m = X.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)

        # Gradient de la couche de sortie (softmax + cross-entropy)
        dZ = outputs - y
        assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        # Rétropropagation dans les couches cachées
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i + 1].T) * relu_derivative(self.z_values[i])
            assert dZ.shape == (X.shape[0], self.layer_sizes[i + 1]), f"dZ^{[i + 1]} has incorrect shape"
            self.d_weights[i] = (self.activations[i].T @ dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        # Bonus : régularisation L2 -> dW^{[l]} += (lambda / m) * W^{[l]}
        if self.lambda_reg > 0:
            for i in range(len(self.weights)):
                self.d_weights[i] += (self.lambda_reg / m) * self.weights[i]

        # Mise à jour des paramètres
        if self.optimizer == "adam":
            self._adam_update()
        else:
            self._sgd_update()

    def _sgd_update(self):
        """Mise à jour classique par descente de gradient."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def _adam_update(self):
        """
        Bonus : mise à jour avec l'optimiseur Adam (moments adaptatifs).
        m = beta1*m + (1-beta1)*dW
        v = beta2*v + (1-beta2)*dW^2
        m_hat = m / (1 - beta1^t) ; v_hat = v / (1 - beta2^t)
        W -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1
        for i in range(len(self.weights)):
            # Poids
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * self.d_weights[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (self.d_weights[i] ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Biais
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * self.d_biases[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.d_biases[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X, y, X_val, y_val, epochs, batch_size, verbose=True):
        """
        Train the neural network using mini-batch SGD/Adam, with validation
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), \
            "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], \
            f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape[1] == self.layer_sizes[-1], \
            f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert X_val.shape[1] == self.layer_sizes[0], \
            f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y_val.shape[1] == self.layer_sizes[-1], \
            f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)

            # Calculer les pertes et accuracies
            train_loss = epoch_loss / (X.shape[0] // batch_size)
            train_pred = self.forward(X)
            train_accuracy = self.compute_accuracy(y, train_pred)
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy(y_val, val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        Predict class labels
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], \
            f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

        outputs = self.forward(X)
        predictions = np.argmax(outputs, axis=1)
        assert predictions.shape == (X.shape[0],), "Predictions have incorrect shape"
        return predictions


# =========================================================
# 3. Chargement des données
# =========================================================
# Chemin corrigé selon la structure réelle du dossier décompressé :
# amazigh-handwritten-character-database-amhcd/amhcd_64/AMHCD_64/<classe>/<image>.png
data_dir = os.path.join(
    os.getcwd(),
    "amazigh-handwritten-character-database-amhcd",
    "amhcd_64",
    "AMHCD_64",
)
print(data_dir)
current_working_directory = os.getcwd()
print(current_working_directory)

# Il n'y a pas de fichier labels-map.csv dans cette version du dataset :
# chaque sous-dossier de AMHCD_64 correspond à une classe (label = nom du dossier).
image_paths = []
labels = []
for label_dir in sorted(os.listdir(data_dir)):
    label_path = os.path.join(data_dir, label_dir)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            image_paths.append(os.path.join(label_path, img_name))
            labels.append(label_dir)
labels_df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Vérifier le DataFrame
assert not labels_df.empty, "No data loaded. Check dataset files."
print(f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")

# Encoder les étiquettes
label_encoder = LabelEncoder()
labels_df["label_encoded"] = label_encoder.fit_transform(labels_df["label"])
num_classes = len(label_encoder.classes_)


def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess an image: convert to grayscale, resize, normalize
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # Normalisation
    return img.flatten()  # Aplatir pour le réseau de neurones


# image_path contient déjà le chemin complet (voir construction ci-dessus)
X = np.array([load_and_preprocess_image(path) for path in labels_df["image_path"]])
y = labels_df["label_encoded"].values

# Vérifier les dimensions
assert X.shape[0] == y.shape[0], "Mismatch between number of images and labels"
assert X.shape[1] == 32 * 32, f"Expected flattened image size of {32 * 32}, got {X.shape[1]}"

# Diviser en ensembles d'entraînement, validation et test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Convertir explicitement en NumPy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0], \
    "Train-val-test split sizes must sum to total samples"

print(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")


# =========================================================
# 4. Bonus : Augmentation de données (rotations + translations)
# =========================================================
def augment_image(flat_img, image_size=32, max_angle=8, max_shift=2):
    """
    Applique une rotation et une translation aléatoires à une image aplatie.
    - max_angle : rotation aléatoire dans [-max_angle, max_angle] degrés
    - max_shift : translation aléatoire dans [-max_shift, max_shift] pixels (x et y)
    """
    img = flat_img.reshape(image_size, image_size)

    angle = np.random.uniform(-max_angle, max_angle)
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)

    center = (image_size / 2, image_size / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_matrix[0, 2] += tx
    rot_matrix[1, 2] += ty

    augmented = cv2.warpAffine(img, rot_matrix, (image_size, image_size),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return augmented.flatten()


def augment_dataset(X_data, y_data, n_augmented_per_sample=1, image_size=32,
                     max_angle=8, max_shift=2, seed=42):
    """
    Génère des versions augmentées (rotation + translation) de chaque échantillon
    et les concatène à l'ensemble d'origine.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    augmented_X = [X_data]
    augmented_y = [y_data]

    for _ in range(n_augmented_per_sample):
        aug_batch = np.array([augment_image(img, image_size, max_angle, max_shift) for img in X_data])
        augmented_X.append(aug_batch)
        augmented_y.append(y_data)

    return np.vstack(augmented_X), np.concatenate(augmented_y)


# Activer/désactiver l'augmentation de données ici
USE_DATA_AUGMENTATION = True
if USE_DATA_AUGMENTATION:
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, n_augmented_per_sample=1)
    print(f"Après augmentation : {X_train_aug.shape[0]} échantillons d'entraînement "
          f"(x{X_train_aug.shape[0] / X_train.shape[0]:.1f})")
else:
    X_train_aug, y_train_aug = X_train, y_train


# =========================================================
# 5. Encodage one-hot
# =========================================================
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = np.array(one_hot_encoder.fit_transform(y_train_aug.reshape(-1, 1)))
y_val_one_hot = np.array(one_hot_encoder.transform(y_val.reshape(-1, 1)))
y_test_one_hot = np.array(one_hot_encoder.transform(y_test.reshape(-1, 1)))

assert isinstance(y_train_one_hot, np.ndarray), "y_train_one_hot must be a numpy array"
assert isinstance(y_val_one_hot, np.ndarray), "y_val_one_hot must be a numpy array"
assert isinstance(y_test_one_hot, np.ndarray), "y_test_one_hot must be a numpy array"


# =========================================================
# 6. Bonus : Validation croisée K-fold (sur train+val, avant l'entraînement final)
# =========================================================
def k_fold_cross_validation(X_data, y_data_encoded, layer_sizes, k=5, epochs=30,
                             batch_size=32, learning_rate=0.01, optimizer="adam",
                             lambda_reg=0.0, num_classes=33, seed=42):
    """
    Évalue la robustesse du modèle avec une validation croisée K-fold.
    Retourne la liste des accuracies de validation pour chaque fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_accuracies = []

    ohe = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_data)):
        X_tr, X_va = X_data[train_idx], X_data[val_idx]
        y_tr, y_va = y_data_encoded[train_idx], y_data_encoded[val_idx]

        y_tr_oh = ohe.fit_transform(y_tr.reshape(-1, 1))
        y_va_oh = ohe.transform(y_va.reshape(-1, 1))

        model = MultiClassNeuralNetwork(layer_sizes, learning_rate=learning_rate,
                                         optimizer=optimizer, lambda_reg=lambda_reg)
        model.train(X_tr, y_tr_oh, X_va, y_va_oh, epochs=epochs,
                    batch_size=batch_size, verbose=False)

        val_pred = model.forward(X_va)
        acc = model.compute_accuracy(y_va_oh, val_pred)
        fold_accuracies.append(acc)
        print(f"Fold {fold_idx + 1}/{k} - Validation Accuracy: {acc:.4f}")

    print(f"\nK-Fold CV — Accuracy moyenne: {np.mean(fold_accuracies):.4f} "
          f"(+/- {np.std(fold_accuracies):.4f})")
    return fold_accuracies


RUN_KFOLD = True
if RUN_KFOLD:
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    layer_sizes_cv = [X_train_full.shape[1], 64, 32, num_classes]
    kfold_accuracies = k_fold_cross_validation(
        X_train_full, y_train_full, layer_sizes_cv,
        k=5, epochs=40, batch_size=32, learning_rate=0.001,
        optimizer="adam", lambda_reg=0.0001, num_classes=num_classes,
    )


# =========================================================
# 7. Créer et entraîner le modèle final
# =========================================================
layer_sizes = [X_train_aug.shape[1], 64, 32, num_classes]  # 64 et 32 neurones cachés, 33 classes
nn = MultiClassNeuralNetwork(
    layer_sizes,
    learning_rate=0.001,      # taux réduit car Adam converge plus vite que le SGD à 0.01
    optimizer="adam",         # bonus : Adam au lieu du SGD classique
    lambda_reg=0.0001,        # bonus : régularisation L2 (réduite pour ne pas freiner le démarrage)
)
train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
    X_train_aug, y_train_one_hot, X_val, y_val_one_hot, epochs=150, batch_size=32
)


# =========================================================
# 8. Prédictions et évaluation
# =========================================================
y_pred = nn.predict(X_test)
print("\nRapport de classification (Test set) :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion (Test set)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Courbes de perte et d'accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Validation Loss")
ax1.set_title("Courbe de perte")
ax1.set_xlabel("Époque")
ax1.set_ylabel("Perte")
ax1.legend()

ax2.plot(train_accuracies, label="Train Accuracy")
ax2.plot(val_accuracies, label="Validation Accuracy")
ax2.set_title("Courbe de précision")
ax2.set_xlabel("Époque")
ax2.set_ylabel("Précision")
ax2.legend()

plt.tight_layout()
fig.savefig("loss_accuracy_plot.png")
plt.close()

print("\nTerminé. Fichiers générés : confusion_matrix.png, loss_accuracy_plot.png")
