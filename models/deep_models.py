"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Deep Learning Models Module

Models:
  - MLP  : Multi-Layer Perceptron (used as meta-learner in stacked ensemble)
  - CNN-BiLSTM : Hybrid deep model for sequential/tabular feature extraction

Built with TensorFlow / Keras.
================================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Define dummy class to prevent NameError in type annotations
    class DummyModel:
        pass
    class DummyCallbacks:
        History = object
    class DummyKeras:
        Model = DummyModel
        callbacks = DummyCallbacks()
    keras = DummyKeras()



def build_mlp(input_dim: int,
              n_classes: int,
              hidden_units: tuple = (256, 128, 64),
              dropout_rate: float = 0.3,
              mc_dropout: bool = True) -> keras.Model:
    """
    Build a Multi-Layer Perceptron classifier.

    Uses Monte Carlo Dropout (MC-Dropout) for uncertainty quantification
    when mc_dropout=True — dropout is active at inference time.

    Parameters
    ----------
    input_dim    : Number of input features
    n_classes    : Number of output classes
    hidden_units : Tuple of neurons per hidden layer
    dropout_rate : Dropout probability
    mc_dropout   : If True, dropout is always active (even at inference)
    """
    inputs = keras.Input(shape=(input_dim,), name='features')
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f'dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        # training=True forces dropout even at inference (MC-Dropout)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x, training=mc_dropout)

    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='MLP_MetaLearner')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_bilstm(input_dim: int,
                     n_classes: int,
                     dropout_rate: float = 0.3) -> keras.Model:
    """
    Build a CNN-BiLSTM hybrid model.

    Architecture:
      Input → Reshape(1D→2D) → Conv1D → MaxPool → BiLSTM → Dense → Softmax

    The 1D feature vector is reshaped into a (features, 1) sequence so
    Conv1D can extract local feature patterns, then BiLSTM captures
    sequential dependencies across the feature space.
    """
    inputs = keras.Input(shape=(input_dim,), name='features')

    # Reshape for Conv1D: (batch, features, 1)
    x = layers.Reshape((input_dim, 1), name='reshape')(inputs)

    # CNN feature extraction
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', name='conv1')(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same', name='conv2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = layers.Dropout(dropout_rate, name='cnn_dropout')(x)

    # BiLSTM sequential learning
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        name='bilstm'
    )(x)
    x = layers.Dropout(dropout_rate, name='lstm_dropout')(x)

    # Classification head
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='fc_dropout')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='CNN_BiLSTM')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_deep_model(model: keras.Model,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     epochs: int = 60,
                     batch_size: int = 64,
                     verbose: int = 0) -> keras.callbacks.History:
    """
    Train a Keras model with early stopping and learning-rate reduction.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


def mc_dropout_predict(model: keras.Model,
                       X: np.ndarray,
                       n_passes: int = 50) -> tuple:
    """
    Monte Carlo Dropout inference for uncertainty quantification.

    Runs `n_passes` stochastic forward passes with dropout active,
    then computes mean and standard deviation of the probability distributions.

    Parameters
    ----------
    model    : Keras model with MC-Dropout
    X        : Input array (n_samples, n_features)
    n_passes : Number of stochastic forward passes

    Returns
    -------
    mean_proba : (n_samples, n_classes) — average probability
    std_proba  : (n_samples, n_classes) — uncertainty (std dev)
    """
    preds = np.stack([model(X, training=True) for _ in range(n_passes)], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)
