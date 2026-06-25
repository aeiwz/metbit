# -*- coding: utf-8 -*-

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

"""PyTorch-based deep learning models for NMR spectral data.

All models accept pandas DataFrames or numpy arrays as input and
internally convert to torch tensors. Requires PyTorch to be installed.
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _torch_err:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for metbit.dl models but is not installed.\n"
        "Install it with:  pip install torch\n"
        "or visit https://pytorch.org for platform-specific instructions.\n"
        f"Original error: {_torch_err}"
    ) from _torch_err

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_numpy(X):
    """Convert DataFrame or array-like to float32 numpy array.

    Args:
        X: Input data as pd.DataFrame, np.ndarray, or array-like.

    Returns:
        np.ndarray with dtype float32.
    """
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    return np.array(X, dtype=np.float32)


def _resolve_device(device: str) -> torch.device:
    """Resolve device string, handling 'auto' selection.

    Args:
        device: One of 'auto', 'cpu', 'cuda', 'mps', or any valid
            torch device string.

    Returns:
        torch.device instance.
    """
    if device == "auto":
        if torch.cuda.is_available():  # pragma: no cover
            return torch.device("cuda")  # pragma: no cover
        if torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")  # pragma: no cover
        return torch.device("cpu")
    return torch.device(device)  # pragma: no cover


def _zscore_normalize(X: np.ndarray):
    """Z-score normalise along features (axis=0).

    Args:
        X: Array of shape (n_samples, n_features).

    Returns:
        Tuple of (X_normalized, mean, std) where std has zeros replaced
        with 1 to avoid division-by-zero.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _default_color_sequence():
    return [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]


# ---------------------------------------------------------------------------
# CLASS 1: SpectralAutoencoder
# ---------------------------------------------------------------------------

class _AutoencoderNet(nn.Module):
    """Internal PyTorch Module for the symmetric autoencoder."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()

        # Build encoder
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder (mirrored hidden dims)
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class SpectralAutoencoder:
    """Symmetric encoder-decoder for unsupervised NMR spectral embedding.

    Trains a fully-connected autoencoder on spectral data and exposes
    methods to retrieve latent embeddings, reconstructions, and
    diagnostic plots.

    Args:
        X: Input spectra as a DataFrame (samples x variables) or ndarray
            of shape (n_samples, n_features).
        latent_dim: Dimensionality of the bottleneck layer.
        hidden_dims: List of hidden layer sizes for the encoder; the
            decoder mirrors these in reverse.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimiser.
        batch_size: Mini-batch size.
        random_state: Seed for reproducibility.
        device: Compute device - 'auto', 'cpu', 'cuda', or 'mps'.
            'auto' selects CUDA > MPS > CPU automatically.

    Attributes:
        training_loss_: List of per-epoch MSE loss values populated after
            calling fit().

    Examples:
        >>> import numpy as np
        >>> from metbit.dl import SpectralAutoencoder
        >>> X = np.random.rand(80, 1000).astype("float32")
        >>> ae = SpectralAutoencoder(X, latent_dim=4, epochs=5)
        >>> ae.fit(verbose=False)
        SpectralAutoencoder(latent_dim=4, ...)
        >>> emb = ae.encode()
        >>> emb.shape
        (80, 4)
    """

    def __init__(
        self,
        X,
        latent_dim: int = 8,
        hidden_dims: list = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "auto",
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        torch.manual_seed(random_state)

        self._X_raw = _to_numpy(X)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = _resolve_device(device)

        # Normalise training data
        self._X_norm, self._mean, self._std = _zscore_normalize(self._X_raw.copy())

        self._input_dim = self._X_norm.shape[1]
        self._model = _AutoencoderNet(self._input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        self.training_loss_: list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def _resolve_input(self, X) -> np.ndarray:
        if X is None:
            return self._X_norm
        arr = _to_numpy(X)
        return self._normalise(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "SpectralAutoencoder":
        """Train the autoencoder.

        Args:
            verbose: If True, print loss every 10 epochs.

        Returns:
            self, to allow method chaining.
        """
        X_tensor = self._to_tensor(self._X_norm)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        self.training_loss_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                out = self._model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            self.training_loss_.append(epoch_loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch [{epoch:>4d}/{self.epochs}]  MSE loss: {epoch_loss:.6f}")

        return self

    def encode(self, X=None) -> np.ndarray:
        """Return latent embeddings for X.

        Args:
            X: Data to encode. If None, encodes the training data.

        Returns:
            np.ndarray of shape (n_samples, latent_dim).
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            z = self._model.encode(self._to_tensor(arr))
        return z.cpu().numpy()

    def reconstruct(self, X=None) -> np.ndarray:
        """Return reconstructed spectra.

        Args:
            X: Data to reconstruct. If None, reconstructs the training
                data.

        Returns:
            np.ndarray of shape (n_samples, n_features) in original
            (un-normalised) scale.
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            out = self._model(self._to_tensor(arr))
        out_np = out.cpu().numpy()
        # Invert z-score
        return out_np * self._std + self._mean

    def plot_embedding(
        self,
        color_: "pd.Series | list | None" = None,
        color_dict: "dict | None" = None,
        components: list = None,
        fig_height: int = 700,
        fig_width: int = 900,
    ) -> go.Figure:
        """Scatter plot of two selected latent dimensions.

        Args:
            color_: Group labels aligned with training samples. Used to
                colour points.
            color_dict: Mapping from label to hex/CSS colour string.
                If None, colours are assigned automatically.
            components: Two-element list of zero-based latent dimension
                indices to plot. Defaults to [0, 1].
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.

        Examples:
            >>> fig = ae.plot_embedding(color_=labels)
            >>> fig.show()
        """
        if components is None:
            components = [0, 1]

        Z = self.encode()
        c0, c1 = components[0], components[1]
        x_vals = Z[:, c0]
        y_vals = Z[:, c1]

        fig = go.Figure()

        if color_ is None:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="markers",
                marker=dict(size=8, color="#636EFA", opacity=0.8),
                name="Samples",
            ))
        else:
            labels = list(color_) if not isinstance(color_, list) else color_
            unique_labels = list(dict.fromkeys(labels))
            palette = _default_color_sequence()
            auto_colors = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
            cmap = color_dict if color_dict is not None else auto_colors

            for lbl in unique_labels:
                idx = [i for i, l in enumerate(labels) if l == lbl]
                fig.add_trace(go.Scatter(
                    x=x_vals[idx], y=y_vals[idx],
                    mode="markers",
                    marker=dict(size=8, color=cmap.get(lbl, "#636EFA"), opacity=0.8),
                    name=str(lbl),
                ))

        fig.update_layout(
            title="Spectral Autoencoder - Latent Embedding",
            xaxis_title=f"Latent dim {c0}",
            yaxis_title=f"Latent dim {c1}",
            height=fig_height,
            width=fig_width,
            template="plotly_white",
            legend_title="Group",
        )
        return fig

    def plot_loss(self, fig_height: int = 400, fig_width: int = 700) -> go.Figure:
        """Plot the training MSE loss curve.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(self.training_loss_) + 1)),
            y=self.training_loss_,
            mode="lines",
            line=dict(color="#636EFA", width=2),
            name="MSE loss",
        ))
        fig.update_layout(
            title="SpectralAutoencoder - Training Loss",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=fig_height,
            width=fig_width,
            template="plotly_white",
        )
        return fig

    def __repr__(self):
        return (
            f"SpectralAutoencoder(latent_dim={self.latent_dim}, "
            f"hidden_dims={self.hidden_dims}, epochs={self.epochs}, "
            f"device='{self.device}')"
        )


# ---------------------------------------------------------------------------
# CLASS 2: SpectralMLP
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """Internal PyTorch Module for the MLP classifier."""

    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpectralMLP:
    """Multi-layer perceptron classifier for NMR spectral data.

    Trains a fully-connected MLP with dropout regularisation using
    cross-entropy loss. Labels are encoded internally via
    sklearn.preprocessing.LabelEncoder.

    Args:
        X: Input spectra as a DataFrame (samples x variables) or ndarray
            of shape (n_samples, n_features).
        y: Class labels as a Series, ndarray, or list. Can be strings or
            integers.
        hidden_dims: List of hidden layer sizes.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimiser.
        batch_size: Mini-batch size.
        dropout: Dropout probability applied after each hidden ReLU.
        random_state: Seed for reproducibility.
        device: Compute device - 'auto', 'cpu', 'cuda', or 'mps'.

    Attributes:
        training_loss_: List of per-epoch cross-entropy loss values
            populated after calling fit().
        label_encoder_: Fitted LabelEncoder instance.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> from metbit.dl import SpectralMLP
        >>> X = np.random.rand(60, 500).astype("float32")
        >>> y = ["ctrl"] * 30 + ["case"] * 30
        >>> clf = SpectralMLP(X, y, epochs=5)
        >>> clf.fit(verbose=False)
        SpectralMLP(hidden_dims=[256, 128, 64], ...)
        >>> preds = clf.predict()
        >>> preds.shape
        (60,)
    """

    def __init__(
        self,
        X,
        y,
        hidden_dims: list = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        dropout: float = 0.3,
        random_state: int = 42,
        device: str = "auto",
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        torch.manual_seed(random_state)

        self._X_raw = _to_numpy(X)
        self._y_raw = np.array(y)
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.device = _resolve_device(device)

        # Normalise
        self._X_norm, self._mean, self._std = _zscore_normalize(self._X_raw.copy())

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        self._y_enc = self.label_encoder_.fit_transform(self._y_raw).astype(np.int64)
        self._n_classes = len(self.label_encoder_.classes_)
        self._input_dim = self._X_norm.shape[1]

        self._model = _MLPNet(
            self._input_dim, self.hidden_dims, self._n_classes, self.dropout
        ).to(self.device)
        self.training_loss_: list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def _to_tensor_X(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def _resolve_input(self, X) -> np.ndarray:
        if X is None:
            return self._X_norm
        arr = _to_numpy(X)
        return self._normalise(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "SpectralMLP":
        """Train the MLP classifier.

        Args:
            verbose: If True, print loss every 10 epochs.

        Returns:
            self, to allow method chaining.
        """
        X_tensor = self._to_tensor_X(self._X_norm)
        y_tensor = torch.tensor(self._y_enc, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        self.training_loss_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)
            self.training_loss_.append(epoch_loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch [{epoch:>4d}/{self.epochs}]  CE loss: {epoch_loss:.6f}")

        return self

    def predict(self, X=None) -> np.ndarray:
        """Return predicted class labels.

        Args:
            X: Data to classify. If None, classifies the training data.

        Returns:
            np.ndarray of class label strings (or original dtype).
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._to_tensor_X(arr))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X=None) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Data to score. If None, scores the training data.

        Returns:
            np.ndarray of shape (n_samples, n_classes).
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._to_tensor_X(arr))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_accuracy(self) -> float:
        """Compute training accuracy.

        Returns:
            Fraction of correctly classified training samples.
        """
        preds = self.predict()
        return float(np.mean(preds == self._y_raw))

    def plot_loss(self, fig_height: int = 400, fig_width: int = 700) -> go.Figure:
        """Plot the training cross-entropy loss curve.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(self.training_loss_) + 1)),
            y=self.training_loss_,
            mode="lines",
            line=dict(color="#EF553B", width=2),
            name="CE loss",
        ))
        fig.update_layout(
            title="SpectralMLP - Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Cross-Entropy Loss",
            height=fig_height,
            width=fig_width,
            template="plotly_white",
        )
        return fig

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        fig_height: int = 600,
        fig_width: int = 700,
    ) -> go.Figure:
        """Plot a confusion matrix heatmap for training predictions.

        Args:
            normalize: If True, show row-normalised proportions;
                otherwise show raw counts.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.
        """
        return _confusion_matrix_fig(
            y_true=self._y_raw,
            y_pred=self.predict(),
            classes=self.label_encoder_.classes_,
            normalize=normalize,
            title="SpectralMLP - Confusion Matrix",
            fig_height=fig_height,
            fig_width=fig_width,
        )

    def __repr__(self):
        return (
            f"SpectralMLP(hidden_dims={self.hidden_dims}, "
            f"epochs={self.epochs}, dropout={self.dropout}, "
            f"device='{self.device}')"
        )


# ---------------------------------------------------------------------------
# CLASS 3: SpectralCNN
# ---------------------------------------------------------------------------

class _CNNNet(nn.Module):
    """Internal PyTorch Module for the 1-D CNN classifier."""

    def __init__(self, input_len: int, filters: list, kernel_size: int,
                 n_classes: int, dropout: float):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in filters:
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1], n_classes)

    def forward(self, x):
        # x: (batch, 1, n_features)
        out = self.conv_blocks(x)
        out = self.pool(out)            # (batch, last_filters, 1)
        out = out.squeeze(-1)           # (batch, last_filters)
        out = self.dropout(out)
        return self.fc(out)


class SpectralCNN:
    """1-D Convolutional Neural Network for NMR spectral classification.

    Each block applies Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d.
    An AdaptiveAvgPool1d(1) aggregates temporal features before the
    final linear classifier.

    Args:
        X: Input spectra as a DataFrame (samples x variables) or ndarray
            of shape (n_samples, n_features).
        y: Class labels as a Series, ndarray, or list.
        filters: Number of filters in each convolutional block.
        kernel_size: Kernel size for all Conv1d layers.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimiser.
        batch_size: Mini-batch size.
        dropout: Dropout probability before the final linear layer.
        random_state: Seed for reproducibility.
        device: Compute device - 'auto', 'cpu', 'cuda', or 'mps'.

    Attributes:
        training_loss_: List of per-epoch cross-entropy loss values
            populated after calling fit().
        label_encoder_: Fitted LabelEncoder instance.

    Examples:
        >>> import numpy as np
        >>> from metbit.dl import SpectralCNN
        >>> X = np.random.rand(60, 1024).astype("float32")
        >>> y = ["ctrl"] * 30 + ["case"] * 30
        >>> cnn = SpectralCNN(X, y, epochs=5)
        >>> cnn.fit(verbose=False)
        SpectralCNN(filters=[32, 64, 128], ...)
        >>> preds = cnn.predict()
        >>> preds.shape
        (60,)
    """

    def __init__(
        self,
        X,
        y,
        filters: list = None,
        kernel_size: int = 7,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        dropout: float = 0.3,
        random_state: int = 42,
        device: str = "auto",
    ):
        if filters is None:
            filters = [32, 64, 128]

        torch.manual_seed(random_state)

        self._X_raw = _to_numpy(X)
        self._y_raw = np.array(y)
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.device = _resolve_device(device)

        # Normalise
        self._X_norm, self._mean, self._std = _zscore_normalize(self._X_raw.copy())

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        self._y_enc = self.label_encoder_.fit_transform(self._y_raw).astype(np.int64)
        self._n_classes = len(self.label_encoder_.classes_)
        self._input_len = self._X_norm.shape[1]

        self._model = _CNNNet(
            self._input_len, self.filters, self.kernel_size, self._n_classes, self.dropout
        ).to(self.device)
        self.training_loss_: list = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def _to_tensor_X(self, X: np.ndarray) -> torch.Tensor:
        # Add channel dimension: (batch, 1, n_features)
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)

    def _resolve_input(self, X) -> np.ndarray:
        if X is None:
            return self._X_norm
        arr = _to_numpy(X)
        return self._normalise(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "SpectralCNN":
        """Train the CNN classifier.

        Args:
            verbose: If True, print loss every 10 epochs.

        Returns:
            self, to allow method chaining.
        """
        X_tensor = self._to_tensor_X(self._X_norm)
        y_tensor = torch.tensor(self._y_enc, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        self.training_loss_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)
            self.training_loss_.append(epoch_loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch [{epoch:>4d}/{self.epochs}]  CE loss: {epoch_loss:.6f}")

        return self

    def predict(self, X=None) -> np.ndarray:
        """Return predicted class labels.

        Args:
            X: Data to classify. If None, classifies the training data.

        Returns:
            np.ndarray of class label strings (or original dtype).
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._to_tensor_X(arr))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X=None) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Data to score. If None, scores the training data.

        Returns:
            np.ndarray of shape (n_samples, n_classes).
        """
        arr = self._resolve_input(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._to_tensor_X(arr))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_accuracy(self) -> float:
        """Compute training accuracy.

        Returns:
            Fraction of correctly classified training samples.
        """
        preds = self.predict()
        return float(np.mean(preds == self._y_raw))

    def plot_loss(self, fig_height: int = 400, fig_width: int = 700) -> go.Figure:
        """Plot the training cross-entropy loss curve.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(self.training_loss_) + 1)),
            y=self.training_loss_,
            mode="lines",
            line=dict(color="#00CC96", width=2),
            name="CE loss",
        ))
        fig.update_layout(
            title="SpectralCNN - Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Cross-Entropy Loss",
            height=fig_height,
            width=fig_width,
            template="plotly_white",
        )
        return fig

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        fig_height: int = 600,
        fig_width: int = 700,
    ) -> go.Figure:
        """Plot a confusion matrix heatmap for training predictions.

        Args:
            normalize: If True, show row-normalised proportions;
                otherwise show raw counts.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.

        Returns:
            plotly.graph_objects.Figure.
        """
        return _confusion_matrix_fig(
            y_true=self._y_raw,
            y_pred=self.predict(),
            classes=self.label_encoder_.classes_,
            normalize=normalize,
            title="SpectralCNN - Confusion Matrix",
            fig_height=fig_height,
            fig_width=fig_width,
        )

    def __repr__(self):
        return (
            f"SpectralCNN(filters={self.filters}, kernel_size={self.kernel_size}, "
            f"epochs={self.epochs}, dropout={self.dropout}, "
            f"device='{self.device}')"
        )


# ---------------------------------------------------------------------------
# Shared plotting helper
# ---------------------------------------------------------------------------

def _confusion_matrix_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    normalize: bool,
    title: str,
    fig_height: int,
    fig_width: int,
) -> go.Figure:
    """Build a Plotly heatmap confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        classes: Ordered class names from LabelEncoder.
        normalize: If True, normalise each row to sum to 1.
        title: Plot title.
        fig_height: Figure height in pixels.
        fig_width: Figure width in pixels.

    Returns:
        plotly.graph_objects.Figure.
    """
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.float64)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
        fmt_text = [[f"{v:.2f}" for v in row] for row in cm]
        cbar_title = "Proportion"
    else:
        fmt_text = [[str(int(v)) for v in row] for row in cm]
        cbar_title = "Count"

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[str(c) for c in classes],
        y=[str(c) for c in classes],
        text=fmt_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title=cbar_title),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        yaxis=dict(autorange="reversed"),
        height=fig_height,
        width=fig_width,
        template="plotly_white",
    )
    return fig
