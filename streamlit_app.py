import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Logistic Regression – Gradient Descent Visualization")

st.markdown("""
This demo shows **Logistic Regression trained using Gradient Descent**.

• Left: **Decision boundary evolution**  
• Right: **Contour plot of logistic loss**  

You can only choose the **polynomial degree** of the model.
""")

# -------------------------------------------------
# Sidebar (ONLY DEGREE)
# -------------------------------------------------
st.sidebar.header("Model Complexity")

degree = st.sidebar.slider(
    "Polynomial Degree",
    min_value=1,
    max_value=4,
    value=1
)

run = st.sidebar.button("▶ Run Gradient Descent")

# -------------------------------------------------
# Dataset (binary classification)
# -------------------------------------------------
X, y = make_classification(
    n_samples=200,
    n_features=1,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    flip_y=0.05,
    random_state=7
)

X = X.flatten()
y = y.astype(float)
n = len(X)

# -------------------------------------------------
# Feature expansion
# -------------------------------------------------
def poly_features(x, degree):
    return np.column_stack([x**i for i in range(degree + 1)])

X_poly = poly_features(X, degree)

# -------------------------------------------------
# Sigmoid & loss
# -------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_hat):
    eps = 1e-9
    return -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )

# -------------------------------------------------
# Initialize parameters
# -------------------------------------------------
theta = np.zeros(degree + 1)
lr = 0.1
steps = 60

# -------------------------------------------------
# Fixed plot limits
# -------------------------------------------------
x_min, x_max = X.min() - 1, X.max() + 1
y_min, y_max = -0.2, 1.2

# -------------------------------------------------
# Contour (only w0 & w1)
# -------------------------------------------------
w0_vals = np.linspace(-10, 10, 100)
w1_vals = np.linspace(-10, 10, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
Z = np.zeros_like(W0)

X_base = poly_features(X, 1)

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        logits = W0[i, j] + W1[i, j] * X
        Z[i, j] = log_loss(y, sigmoid(logits))

# -------------------------------------------------
# Layout
# -------------------------------------------------
col1, col2 = st.columns(2)
boundary_plot = col1.empty()
contour_plot = col2.empty()

# -------------------------------------------------
# Gradient Descent Animation
# -------------------------------------------------
if run:

    for step in range(steps):

        # ---------- Forward ----------
        logits = X_poly @ theta
        y_hat = sigmoid(logits)

        # ---------- Gradient ----------
        grad = (1 / n) * X_poly.T @ (y_hat - y)
        theta -= lr * grad

        # ---------- LEFT: Decision boundary ----------
        fig1, ax1 = plt.subplots()

        ax1.scatter(X[y == 0], y[y == 0],
                    color="blue", label="Class 0")
        ax1.scatter(X[y == 1], y[y == 1],
                    color="red", label="Class 1")

        x_plot = np.linspace(x_min, x_max, 300)
        Xp = poly_features(x_plot, degree)
        y_prob = sigmoid(Xp @ theta)

        ax1.plot(x_plot, y_prob,
                 color="black", linewidth=2)

        ax1.axhline(0.5, linestyle="--", color="gray")
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"Decision Boundary – Step {step+1}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("P(y=1)")
        ax1.legend()

        boundary_plot.pyplot(fig1)

        # ---------- RIGHT: Contour ----------
        fig2, ax2 = plt.subplots()
        ax2.contour(W0, W1, Z, levels=30, cmap="viridis")
        ax2.scatter(theta[0], theta[1], color="red", s=60)

        ax2.set_title("Logistic Loss Contour (w0, w1)")
        ax2.set_xlabel("w0 (bias)")
        ax2.set_ylabel("w1 (weight)")

        contour_plot.pyplot(fig2)

        time.sleep(0.25)
