from pathlib import Path

import numpy as np


def main():
    base = Path("../data/input")
    x_path = base / "x_data.npy"
    y_path = base / "y_data.npy"

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    print("\nFirst 5 y values:", y[:5])
    print("\nFirst row of X:", X[0])
    print("\nFirst 5 rows of X:")
    print(X[:5])


if __name__ == "__main__":
    main()
