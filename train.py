import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# split 60/20/20 with class balance
def stratified_split(X, y, rng, train_frac=0.6, val_frac=0.2):
    classes = np.unique(y)
    idx_train, idx_val, idx_test = [], [], []
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_train = int(train_frac * n)
        n_val   = int(val_frac  * n)
        idx_train.extend(idx_c[:n_train])
        idx_val.extend(idx_c[n_train:n_train+n_val])
        idx_test.extend(idx_c[n_train+n_val:])
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

# sigmoid
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

# binary cross-entropy
def binaryCE(y, probs, eps=1e-12):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

# gradient descent (with L2 on weights, not bias)
def gradientDecent(X, y):
    alpha = 0.1
    iters = 3000
    lam = 0.01
    theta = np.zeros(X.shape[1])
    history = []
    for t in range(iters):
        probs = sigmoid(X @ theta)
        grad = (X.T @ (probs - y)) / X.shape[0]
        reg = theta.copy(); reg[0] = 0.0
        grad += lam * reg
        theta -= alpha * grad
        if t % 100 == 0:
            loss = binaryCE(y, probs) + 0.5 * lam * np.sum(reg ** 2)
            history.append((t, loss))
    print("theta:", theta)
    return history, theta

# plot loss
def plotCurve(history):
    iters = [x[0] for x in history]
    losses = [x[1] for x in history]
    plt.plot(iters, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss (Logistic Regression)')
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

# load and split
X_all, y_all = load_breast_cancer(return_X_y=True)
y_all = y_all.astype(int)
rng = np.random.default_rng(42)
idx_train, idx_val, idx_test = stratified_split(X_all, y_all, rng)

# standardize using train mean/std, then add bias = 1
mean = X_all[idx_train].mean(axis=0)
std  = X_all[idx_train].std(axis=0); std[std == 0] = 1.0

X_train = (X_all[idx_train] - mean) / std
X_test  = (X_all[idx_test]  - mean) / std

X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
X_test  = np.column_stack([np.ones(X_test.shape[0]),  X_test])

y_train = y_all[idx_train]
y_test  = y_all[idx_test]

# train
history, theta = gradientDecent(X_train, y_train)
plotCurve(history)

# test predictions
probs = sigmoid(X_test @ theta)
preds = (probs >= 0.5).astype(int)

# confusion counts
true_positive  = int(np.sum((y_test == 1) & (preds == 1)))
true_negative  = int(np.sum((y_test == 0) & (preds == 0)))
false_positive = int(np.sum((y_test == 0) & (preds == 1)))
false_negative = int(np.sum((y_test == 1) & (preds == 0)))

accuracy  = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-12)
precision = true_positive / (true_positive + false_positive + 1e-12)
recall    = true_positive / (true_positive + false_negative + 1e-12)
f1_score  = 2 * precision * recall / (precision + recall + 1e-12)

print("\n=== Test results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1_score:.4f}")
print("Counts [TP, TN, FP, FN]:", [true_positive, true_negative, false_positive, false_negative])

print("\nFirst 5 test examples (prob, pred, true):")
for i in range(min(5, len(y_test))):
    print(f"{i}: {probs[i]:.6f}\t{int(preds[i])}\t{int(y_test[i])}")
