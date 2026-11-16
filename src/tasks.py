from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

import keras

sns.set_style('whitegrid')

RANDOM_STATE = 42
TARGET = 'quality'
CATEGORICAL = 'type'
QUALITY_BINS = (-np.inf, 5, 7, np.inf)
CLASS_NAMES = ('Low', 'Medium', 'High')
HIDDEN_UNITS = (64, 32, 16)


def _read_wine_csv(path: str | Path, wine_type: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df.assign(**{CATEGORICAL: wine_type})


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and combining datasets of red and white wine.

    Note: Don't forget to add the 'type' characteristic
    """
    red = _read_wine_csv(red_path, 'red')
    white = _read_wine_csv(white_path, 'white')

    data = pd.concat([red, white], join='inner', ignore_index=True)
    feature_cols = data.columns.drop([TARGET, CATEGORICAL])
    data[feature_cols] = data[feature_cols].apply(pd.to_numeric,
                                                  errors='coerce')
    data[TARGET] = pd.to_numeric(data[TARGET], errors='coerce')

    return (data.dropna()
            .drop_duplicates()
            .astype({TARGET: int})
            .reset_index(drop=True))


def remove_outliers(df: pd.DataFrame,
                    features: pd.Index | pd.DataFrame) -> pd.DataFrame:
    """Removing outliers from the dataset using the IQR method."""
    columns = features.columns if isinstance(features,
                                             pd.DataFrame) else features
    df_clean = df.copy()

    q1 = df_clean[columns].quantile(0.25)
    q3 = df_clean[columns].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    mask = df_clean[columns].ge(lower).all(axis=1) & df_clean[columns].le(
        upper).all(axis=1)

    return df_clean.loc[mask].reset_index(drop=True)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Standardization of features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding categorical variables.

    This time you need to use LabelEncoder
    """
    data_encoded = data.copy()
    data_encoded[CATEGORICAL] = LabelEncoder().fit_transform(
        data_encoded[CATEGORICAL])

    return data_encoded


def prepare_regression_data(
        data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preparing data for the regression task."""
    X = data.drop(columns=[TARGET, CATEGORICAL])
    y = data[TARGET]

    return X, y


def _build_dense_network(input_dim: int, output_units: int,
                         output_activation: str | None) -> keras.Sequential:
    return keras.Sequential([
        keras.layers.Input(shape=(input_dim, )),
        keras.layers.Dense(input_dim, activation='relu'),
        *(keras.layers.Dense(units, activation='relu')
          for units in HIDDEN_UNITS),
        keras.layers.Dense(output_units, activation=output_activation),
    ])


def build_regression_model(input_dim: int) -> keras.Model:
    """
    Creating a neural network regressor.

    Use `keras.layers.Input` and `keras.layers.Dense`.
    """
    model = _build_dense_network(input_dim, 1, None)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model


def _fit(model: keras.Model, X_train, y_train, epochs: int,
         batch_size: int) -> keras.callbacks.History:
    return model.fit(X_train,
                     y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=0.2,
                     verbose=1)


def train_regression_model(model: keras.Model,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           epochs: int = 50,
                           batch_size: int = 32) -> keras.callbacks.History:
    """Training the neural network regressor."""
    return _fit(model, X_train, y_train, epochs, batch_size)


def evaluate_regression_model(
        model: keras.Model, X_test: pd.DataFrame,
        y_test: pd.Series) -> dict[str, float | np.ndarray]:
    """Evaluating the neural network regressor."""
    y_pred = model.predict(X_test, verbose=0).ravel()

    return {
        'MSE': float(mean_squared_error(y_test, y_pred)),
        'R2': float(r2_score(y_test, y_pred)),
    }


def prepare_classification_data(
        data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Preparing data for the classification task.

    Use keras.utils.to_categorical to convert labels
    into a one-hot encoded vector.
    """
    labels = pd.cut(data[TARGET],
                    bins=QUALITY_BINS,
                    labels=range(len(CLASS_NAMES))).astype(int)
    data = data.assign(quality_label=labels)

    X = data.drop(columns=[TARGET, 'quality_label'])
    y = keras.utils.to_categorical(data['quality_label'],
                                   num_classes=len(CLASS_NAMES))

    return X, y


def build_classification_model(input_dim: int,
                               num_classes: int) -> keras.Model:
    """
    Creating a neural network classifier.

    Use `keras.layers.Input` and `keras.layers.Dense`.
    """
    model = _build_dense_network(input_dim, num_classes, 'softmax')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_classification_model(model: keras.Model,
                               X_train: pd.DataFrame,
                               y_train: np.ndarray,
                               epochs: int = 50,
                               batch_size: int = 32) -> keras.callbacks.History:
    """Training the neural network classifier."""
    return _fit(model, X_train, y_train, epochs, batch_size)


def evaluate_classification_model(
        model: keras.Model, X_test: pd.DataFrame,
        y_test: np.ndarray) -> dict[str, float | str | dict]:
    """Evaluating the neural network classifier."""
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    y_true = np.asarray(y_test).argmax(axis=1)
    labels = range(len(CLASS_NAMES))

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'report': classification_report(y_true,
                                        y_pred,
                                        labels=labels,
                                        target_names=CLASS_NAMES,
                                        zero_division=0),
    }


def main():
    keras.utils.set_random_seed(RANDOM_STATE)

    # Loading data
    data = load_data('winequality-red.csv', 'winequality-white.csv')

    # Removing outliers
    features = data.columns[:-2]  # Without 'quality' and 'type'
    data = remove_outliers(data, features)

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Regression task
    X_reg, y_reg = prepare_regression_data(data)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)

    # Feature scaling
    X_train_scaled_reg, X_test_scaled_reg = scale_features(
        X_train_reg, X_test_reg)

    reg_model = build_regression_model(X_train_scaled_reg.shape[1])
    train_regression_model(reg_model, X_train_scaled_reg, y_train_reg)
    regression_results = evaluate_regression_model(reg_model,
                                                   X_test_scaled_reg,
                                                   y_test_reg)
    print("Regression results:")
    print(f"MSE - {regression_results['MSE']:.4f}")
    print(f"R2 - {regression_results['R2']:.4f}")

    # Classification task
    X_clf, y_clf = prepare_classification_data(data)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf.argmax(axis=1))

    # Feature scaling
    X_train_scaled_clf, X_test_scaled_clf = scale_features(
        X_train_clf, X_test_clf)

    clf_model = build_classification_model(X_train_scaled_clf.shape[1],
                                           y_clf.shape[1])
    train_classification_model(clf_model, X_train_scaled_clf, y_train_clf)

    classification_results = evaluate_classification_model(
        clf_model, X_test_scaled_clf, y_test_clf)
    print("\nClassification results:")
    print(f"Accuracy: {classification_results['accuracy']:.4f}")
    print("Classification report:")
    print(classification_results['report'])


if __name__ == "__main__":
    main()
