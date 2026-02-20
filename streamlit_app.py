import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Polynomial Regression – Gradient Descent Visualization")

st.markdown("""
**Polynomial Regression using Gradient Descent (from scratch)**

• Left: Polynomial curve evolving step-by-step  
• Right: Contour plot of loss J(w₀, w₁)  
• You can increase degree **up to 1000**  
""")

# -------------------------------------------------
# Sidebar (ONLY DEGREE)
# -------------------------------------------------
st.sidebar.header("Model Complexity")

degree = st.sidebar.slider(
    "Polynomial Degree",
    min_value=1,
    max_value=1000,
    value=2
)

run = st.sidebar.button("▶ Run Gradient Descent")

# -------------------------------------------------
# Dataset (EXACTLY as requested)
# y = 0.8x² + 0.9x + 2 + noise
# -------------------------------------------------
np.random.seed(42)
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

X = X.flatten()
y = y.flatten()
n = len(X)

# -------------------------------------------------
# Normalize X (CRITICAL for high degree)
# -------------------------------------------------
X_norm = X / np.max(np.abs(X))

# -------------------------------------------------
# Polynomial features
# -------------------------------------------------
def poly_features(x, degree):
    return np.column_stack([x**i for i in range(degree + 1)])

X_poly = poly_features(X_norm, degree)

# -------------------------------------------------
# Initialize parameters
# -------------------------------------------------
theta = np.zeros(degree + 1)

lr = 0.01
steps = 60

# -------------------------------------------------
# Loss
# -------------------------------------------------
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# -------------------------------------------------
# Fixed plotting limits
# -------------------------------------------------
x_min, x_max = X.min() - 0.5, X.max() + 0.5
y_min, y_max = y.min() - 5, y.max() + 5

# -------------------------------------------------
# Contour for (w0, w1) ONLY
# -------------------------------------------------
w0_vals = np.linspace(-10, 10, 100)
w1_vals = np.linspace(-10, 10, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
Z = np.zeros_like(W0)

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        y_hat = W0[i, j] + W1[i, j] * X_norm
        Z[i, j] = mse(y, y_hat)

# -------------------------------------------------
# Layout
# -------------------------------------------------
col1, col2 = st.columns(2)
curve_plot = col1.empty()
contour_plot = col2.empty()

# -------------------------------------------------
# Gradient Descent Animation
# -------------------------------------------------
if run:

    for step in range(steps):

        # ---------- Forward ----------
        y_hat = X_poly @ theta

        # ---------- Gradient ----------
        grad = (2 / n) * X_poly.T @ (y_hat - y)
        theta -= lr * grad

        # ---------- LEFT: Polynomial curve ----------
        fig1, ax1 = plt.subplots()

        ax1.scatter(X, y, color="blue", alpha=0.6, label="Data")

        x_plot = np.linspace(x_min, x_max, 400)
        x_plot_norm = x_plot / np.max(np.abs(X))
        X_plot_poly = poly_features(x_plot_norm, degree)

        y_plot = X_plot_poly @ theta

        ax1.plot(x_plot, y_plot, color="red", linewidth=3)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"Polynomial Curve – Step {step+1}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("y")
        ax1.legend()

        curve_plot.pyplot(fig1)

        # ---------- RIGHT: Contour ----------
        fig2, ax2 = plt.subplots()
        ax2.contour(W0, W1, Z, levels=30, cmap="viridis")
        ax2.scatter(theta[0], theta[1], color="red", s=60)

        ax2.set_title("Loss Contour (w₀, w₁)")
        ax2.set_xlabel("w₀ (bias)")
        ax2.set_ylabel("w₁ (linear weight)")

        contour_plot.pyplot(fig2)

        time.sleep(0.25)
