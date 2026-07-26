"""
Three model families, deliberately of increasing complexity:

  get_random_forest()  -- sklearn, tabular baseline
  get_xgboost()         -- gradient boosting, still tabular but usually
                            the strongest of the tabular options
  DepthWindowCNN         -- a 1D conv net over a depth window, written
                            by hand (forward + backward pass, no
                            autograd) so it can actually see bedding
                            context instead of treating every depth
                            as an independent row.

The CNN is small on purpose. With ~2800 training rows across 9 very
imbalanced classes, a deep net will just memorize -- one conv layer
feeding a small MLP head is about the ceiling of what this dataset
can support.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight


def get_random_forest(random_state=13):
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def get_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
    )


def xgb_sample_weights(y):
    return compute_sample_weight(class_weight="balanced", y=y)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class DepthWindowCNN:
    """1D convolution over a depth window (shape N x window x n_curves),
    one conv layer + one hidden dense layer + softmax output. Trained
    with plain mini-batch SGD and momentum, weighted cross-entropy to
    deal with the facies imbalance. Everything below is hand-derived,
    not a wrapper around an autograd framework -- forward/backward are
    both explicit.
    """

    def __init__(self, window, n_curves, n_classes, n_filters=12,
                 kernel_size=3, hidden_dim=48, seed=7):
        rng = np.random.default_rng(seed)
        self.K, self.C_in, self.F = kernel_size, n_curves, n_filters
        self.out_len = window - kernel_size + 1
        self.n_classes = n_classes

        # He init, scaled by fan-in
        self.W_conv = rng.normal(0, np.sqrt(2 / (kernel_size * n_curves)),
                                  size=(kernel_size, n_curves, n_filters))
        self.b_conv = np.zeros(n_filters)

        flat_dim = self.out_len * n_filters
        self.W1 = rng.normal(0, np.sqrt(2 / flat_dim), size=(flat_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)

        # momentum buffers
        self._m = {k: np.zeros_like(getattr(self, k))
                   for k in ("W_conv", "b_conv", "W1", "b1", "W2", "b2")}

    def _im2col(self, X):
        # X: (N, L, C_in) -> (N, out_len, K*C_in)
        N, L, C = X.shape
        cols = np.empty((N, self.out_len, self.K * C))
        for i in range(self.out_len):
            cols[:, i, :] = X[:, i:i + self.K, :].reshape(N, self.K * C)
        return cols

    def forward(self, X):
        N = X.shape[0]
        cols = self._im2col(X)
        Wc_flat = self.W_conv.reshape(self.K * self.C_in, self.F)
        conv_out = cols @ Wc_flat + self.b_conv          # (N, out_len, F)
        conv_relu = np.maximum(conv_out, 0)
        flat = conv_relu.reshape(N, -1)
        z1 = flat @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        probs = _softmax(z2)
        cache = dict(X=X, cols=cols, conv_out=conv_out, conv_relu=conv_relu,
                     flat=flat, z1=z1, a1=a1)
        return probs, cache

    def backward(self, probs, y_idx, cache, sample_w, lr, momentum=0.9):
        N = probs.shape[0]
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), y_idx] = 1.0
        w = sample_w.reshape(-1, 1)

        dz2 = w * (probs - y_onehot) / N
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["flat"].T @ dz1
        db1 = dz1.sum(0)

        dflat = dz1 @ self.W1.T
        dconv_relu = dflat.reshape(cache["conv_relu"].shape)
        dconv_out = dconv_relu * (cache["conv_out"] > 0)

        Wc_flat_shape = (self.K * self.C_in, self.F)
        dWc = np.zeros(Wc_flat_shape)
        cols = cache["cols"]
        for i in range(self.out_len):
            dWc += cols[:, i, :].T @ dconv_out[:, i, :]
        db_conv = dconv_out.sum(axis=(0, 1))

        grads = dict(W_conv=dWc.reshape(self.W_conv.shape), b_conv=db_conv,
                     W1=dW1, b1=db1, W2=dW2, b2=db2)
        for k, g in grads.items():
            self._m[k] = momentum * self._m[k] + g
            setattr(self, k, getattr(self, k) - lr * self._m[k])

    def predict_proba(self, X, batch_size=512):
        out = []
        for i in range(0, len(X), batch_size):
            p, _ = self.forward(X[i:i + batch_size])
            out.append(p)
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _snapshot(self):
        return {k: getattr(self, k).copy() for k in
                ("W_conv", "b_conv", "W1", "b1", "W2", "b2")}

    def _restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, v)

    def fit(self, X, y_idx, sample_weight, X_val=None, y_val_idx=None,
            epochs=60, batch_size=64, lr=0.02, lr_decay=0.97, seed=0,
            verbose=False, restore_best=True):
        rng = np.random.default_rng(seed)
        n = len(X)
        history = {"train_acc": [], "val_acc": []}
        best_val, best_snap, best_epoch = -1, None, -1
        for epoch in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = order[i:i + batch_size]
                probs, cache = self.forward(X[idx])
                self.backward(probs, y_idx[idx], cache, sample_weight[idx], lr)
            lr *= lr_decay

            train_acc = (self.predict(X) == y_idx).mean()
            history["train_acc"].append(train_acc)
            if X_val is not None:
                val_acc = (self.predict(X_val) == y_val_idx).mean()
                history["val_acc"].append(val_acc)
                if val_acc > best_val:
                    best_val, best_snap, best_epoch = val_acc, self._snapshot(), epoch
                if verbose and epoch % 10 == 0:
                    print(f"epoch {epoch:3d}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")
            elif verbose and epoch % 10 == 0:
                print(f"epoch {epoch:3d}  train_acc {train_acc:.3f}")

        if restore_best and best_snap is not None:
            self._restore(best_snap)
            history["best_epoch"] = best_epoch
            history["best_val_acc"] = best_val
            if verbose:
                print(f"restored weights from epoch {best_epoch} (val_acc {best_val:.3f})")
        return history
