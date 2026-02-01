import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# Page setup
# ==================================================
st.set_page_config(page_title="Neural Networks: Visual Intuition", layout="wide")

st.title("🧠 Neural Networks — Visual Intuition")
st.markdown("From decisions → loss → gradients → learning")

# ==================================================
# Utility functions
# ==================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def mae(e):
    return np.abs(e)

def mse(e):
    return e**2

# ==================================================
# PART 1: DECISION GEOMETRY
# ==================================================
st.header("🧮 Part 1: A Single Neuron Makes Decisions")

col1, col2 = st.columns([1.2, 1.0])

with col1:
    gate = st.radio("Logic task", ["AND", "OR"], horizontal=True)

    x1 = st.radio("x₁", [0, 1], horizontal=True)
    x2 = st.radio("x₂", [0, 1], horizontal=True)

    w1 = st.slider("w₁", -3.0, 3.0, 0.5)
    w2 = st.slider("w₂", -3.0, 3.0, 0.5)
    b  = st.slider("b",  -3.0, 3.0, -0.5)

    activation = st.radio(
        "Activation",
        ["No activation (z)", "Sigmoid", "Tanh"]
    )

target = (x1 & x2) if gate == "AND" else (x1 | x2)

with col2:
    xx, yy = np.meshgrid(
        np.linspace(-0.3, 1.3, 250),
        np.linspace(-0.3, 1.3, 250)
    )

    z_bg = w1 * xx + w2 * yy + b

    if activation == "No activation (z)":
        values_bg = z_bg
        threshold = 0.0
        out = w1*x1 + w2*x2 + b
        title = "Linear neuron"
    elif activation == "Sigmoid":
        values_bg = sigmoid(z_bg)
        threshold = st.slider("Sigmoid threshold", 0.0, 1.0, 0.5)
        out = sigmoid(w1*x1 + w2*x2 + b)
        title = "Sigmoid neuron"
    else:
        values_bg = tanh(z_bg)
        threshold = st.slider("Tanh threshold", -1.0, 1.0, 0.0)
        out = tanh(w1*x1 + w2*x2 + b)
        title = "Tanh neuron"

    region = values_bg > threshold

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.contourf(xx, yy, region,
                levels=[-0.5, 0.5, 1.5],
                colors=["lightgray", "white"])

    if abs(w2) > 1e-6:
        xs = np.linspace(-0.3, 1.3, 200)
        ys = (-w1 * xs - b) / w2
        ax.plot(xs, ys, "k--", linewidth=2)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_true = np.array([
        (p[0] & p[1]) if gate == "AND" else (p[0] | p[1])
        for p in X
    ])
    colors = ["green" if y else "red" for y in y_true]

    ax.scatter(X[:,0], X[:,1], c=colors, s=110,
               edgecolors="black")
    ax.scatter(x1, x2, s=220, facecolors="none",
               edgecolors="blue", linewidths=2)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(title)

    st.pyplot(fig)

    st.markdown(f"""
    **Output:** `{out:.2f}`  
    **Prediction:** `{int(out > threshold)}`  
    **Correct:** `{target}`
    """)

# ==================================================
# PART 2: LOSS FUNCTIONS
# ==================================================
st.header("📉 Part 2: Loss Functions")

prediction = st.slider("Prediction", -1.5, 1.5, 0.2, 0.05)
target_val = st.slider("Target", -1.5, 1.5, 1.0, 0.05)

error = prediction - target_val

e = np.linspace(-2, 2, 400)

fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=100)

ax.plot(e, mae(e), label="MAE", linewidth=2)
ax.plot(e, mse(e), label="MSE", linewidth=2)

ax.scatter(error, mae(error), color="red", zorder=3)
ax.scatter(error, mse(error), color="blue", zorder=3)

ax.axvline(0, linestyle="--", color="gray", alpha=0.6)

ax.set_title("Loss vs Error", fontsize=11)
ax.set_xlabel("Error")
ax.set_ylabel("Loss")

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout(pad=0.6)
st.pyplot(fig, use_container_width=False)

# ==================================================
# PART 2b: GRADIENT INTUITION
# ==================================================
st.header("📐 Part 2b: Gradient = Slope")

Δ = 0.4
x2_g = st.slider("Move along error axis", -1.5, 1.5, 0.6, 0.05)
x1_g = x2_g - Δ

col1, col2 = st.columns(2)

for name, fn, col in [("MAE", mae, col1), ("MSE", mse, col2)]:
    y1, y2 = fn(x1_g), fn(x2_g)
    slope = (y2 - y1) / Δ

    with col:
        fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=100)

        ax.plot(e, mae(e), label="MAE", linewidth=2)
        ax.plot(e, mse(e), label="MSE", linewidth=2)

        ax.scatter(error, mae(error), color="red", zorder=3)
        ax.scatter(error, mse(error), color="blue", zorder=3)

        ax.axvline(0, linestyle="--", color="gray", alpha=0.6)

        ax.set_title("Loss vs Error", fontsize=11)
        ax.set_xlabel("Error")
        ax.set_ylabel("Loss")

        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout(pad=0.6)
        st.pyplot(fig)

# ==================================================
# PART 3: WEIGHT UPDATE WITH GRADIENTS
# ==================================================
st.header("🧠 Part 3: Learning = Weight Updates")

# --- Session state ---
if "W" not in st.session_state:
    st.session_state.W = {
        "w1": np.random.uniform(-1, 1),
        "w2": np.random.uniform(-1, 1),
        "b":  np.random.uniform(-1, 1),
        "losses": []
    }

W = st.session_state.W

col1, col2 = st.columns([1.4, 1.0])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("📐 Model, Loss, and Gradients")

    x1 = st.radio("x₁ (train)", [0,1], horizontal=True, key="p3x1")
    x2 = st.radio("x₂ (train)", [0,1], horizontal=True, key="p3x2")
    t  = st.radio("Target", [0,1], horizontal=True, key="p3t")

    loss_type = st.radio("Loss function", ["MSE", "MAE"])
    lr = st.slider("Learning rate", 0.01, 1.0, 0.1)

    # Forward
    z = W["w1"]*x1 + W["w2"]*x2 + W["b"]
    y = sigmoid(z)
    error = y - t

    # Loss
    if loss_type == "MSE":
        loss = error**2
        dL_dy = 2*error
        loss_formula = r"(y - t)^2"
    else:
        loss = abs(error)
        dL_dy = np.sign(error)
        loss_formula = r"|y - t|"

    # Gradients
    dy_dz = y*(1-y)
    dL_dw1 = dL_dy * dy_dz * x1
    dL_dw2 = dL_dy * dy_dz * x2
    dL_db  = dL_dy * dy_dz

    st.markdown("### Weights & Bias")
    st.markdown(f"""
    w₁ = `{W['w1']:.3f}`  
    w₂ = `{W['w2']:.3f}`  
    b  = `{W['b']:.3f}`
    """)

    st.markdown("### Loss")
    st.latex(f"L = {loss_formula}")
    st.markdown(f"**Loss value:** `{loss:.4f}`")

    st.markdown("### Gradients")
    st.latex(r"\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial z}x_1")
    st.markdown(f"`∂L/∂w₁ = {dL_dw1:.4f}`")

    st.latex(r"\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial z}x_2")
    st.markdown(f"`∂L/∂w₂ = {dL_dw2:.4f}`")

    st.latex(r"\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial z}")
    st.markdown(f"`∂L/∂b = {dL_db:.4f}`")

    if st.button("▶️ Update (one step)"):
        W["w1"] -= lr * dL_dw1
        W["w2"] -= lr * dL_dw2
        W["b"]  -= lr * dL_db
        W["losses"].append(loss)

    if st.button("🔄 Reset"):
        W["w1"] = np.random.uniform(-1, 1)
        W["w2"] = np.random.uniform(-1, 1)
        W["b"]  = np.random.uniform(-1, 1)
        W["losses"] = []

# ---------------- RIGHT PANEL ----------------
with col2:
    xx, yy = np.meshgrid(
        np.linspace(-0.3, 1.3, 250),
        np.linspace(-0.3, 1.3, 250)
    )
    z_bg = W["w1"]*xx + W["w2"]*yy + W["b"]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.contourf(xx, yy, z_bg > 0,
                levels=[-0.5, 0.5, 1.5],
                colors=["lightgray", "white"])

    if abs(W["w2"]) > 1e-6:
        xs = np.linspace(-0.3, 1.3, 200)
        ys = (-W["w1"]*xs - W["b"]) / W["w2"]
        ax.plot(xs, ys, "k--", linewidth=2)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_true = np.array([
        (p[0] & p[1]) if gate == "AND" else (p[0] | p[1])
        for p in X
    ])
    colors = ["green" if y else "red" for y in y_true]

    ax.scatter(X[:,0], X[:,1], c=colors,
               edgecolors="black", s=110)
    ax.scatter(x1, x2, s=260, facecolors="none",
               edgecolors="blue", linewidths=2)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title("Decision Region")

    st.pyplot(fig)

    if W["losses"]:
        fig2, ax2 = plt.subplots(figsize=(4.6, 2.6))
        ax2.plot(W["losses"], marker="o")
        ax2.set_title("Loss History")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        st.pyplot(fig2)

# ==========================
# Learning Rate Intuition Demo (Double Well)
# ==========================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.markdown("## Learning Rate Impact: Double-Well Loss Landscape")

st.markdown(
    """
This demo shows how learning rate affects optimization dynamics.
A ball performs gradient descent in a double-well loss function.
"""
)

# ---- Loss definition ----
def loss(x):
    return (x**2 - 1)**2

def grad_loss(x):
    return 4 * x * (x**2 - 1)

# ---- Controls ----
col1, col2 = st.columns(2)

with col1:
    lr = st.slider(
        "Learning rate",
        min_value=0.001,
        max_value=1.0,
        value=0.05,
        step=0.001
    )

with col2:
    steps = st.slider(
        "Optimization steps",
        min_value=20,
        max_value=300,
        value=120,
        step=10
    )

x0 = st.slider(
    "Initial position",
    min_value=-2.0,
    max_value=2.0,
    value=-1.5,
    step=0.1
)

animate = st.checkbox("Animate", value=True)

# ---- Prepare landscape ----
x_space = np.linspace(-2.2, 2.2, 400)
y_space = loss(x_space)

# ---- Optimization loop ----
x = x0
trajectory = [x]

for _ in range(steps):
    g = grad_loss(x)
    g = np.clip(g, -10.0, 10.0)
    x = x - lr * g
    x = np.clip(x, -3.0, 3.0)
    trajectory.append(x)

# ---- Plot ----
plot_area = st.empty()

# ---- Plot ----
plot_area = st.empty()

FIGSIZE = (3.2, 2.6)
DPI = 90

if animate:
    for i, x_i in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        ax.plot(x_space, y_space, linewidth=2)
        ax.scatter(x_i, loss(x_i), s=45, zorder=3)

        ax.set_xlabel("x")
        ax.set_ylabel("Loss")
        ax.set_title(f"Step {i} | LR = {lr:.3f}", fontsize=10)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(pad=0.6)

        plot_area.pyplot(fig, use_container_width=False)
        plt.close(fig)
        time.sleep(0.04)
else:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(x_space, y_space, linewidth=2)
    ax.plot(
        trajectory,
        loss(np.array(trajectory)),
        marker="o",
        markersize=3,
        linewidth=1
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning Rate = {lr:.3f}", fontsize=10)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.6)
    plot_area.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---- Interpretation ----
st.markdown(
    """
### Interpretation
- **Very small LR** → ball crawls, stays in one well.
- **Moderate LR** → fast convergence to a minimum.
- **Large LR** → overshooting causes oscillation between wells.
  
This is the same instability–plasticity tradeoff seen in neural network training.
"""
)
