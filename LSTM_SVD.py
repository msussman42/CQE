#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM_ODE.py
------------
This script trains an LSTM to learn the flow map of a damped oscillator ODE.
It saves all artifacts (Keras/ONNX models, history, evaluation plots)
and can be visualized in https://netron.app.

───────────────────────────────────────────────────────────────
Tested & Compatible Environment (Stable Stack)
───────────────────────────────────────────────────────────────
- OS: Linux / macOS / Windows
- Python: 3.10 – 3.12
- TensorFlow: 2.16.1
- Keras: 3.3+ (bundled in TF 2.16.1)
- tf2onnx: 1.16.1
- onnx: 1.16.0
- onnxruntime: 1.19.2
- protobuf: 3.20.3
───────────────────────────────────────────────────────────────
Optional (for visualization & diagrams):
- netron (view neural networks)
- graphviz + pydot (for saving architecture diagrams)
───────────────────────────────────────────────────────────────

1. Create and activate a virtual environment
------------------------------------------------
# Linux / macOS
python3 -m venv venv_sussman
source venv_sussman/bin/activate

# Windows (PowerShell)
python -m venv venv_sussman
venv_sussman\Scripts\activate

2. Install all required packages (Stable Stack)
--------------------------------------------------
# Recommended compatible versions
pip install --upgrade pip
pip install "tensorflow==2.16.1" "tensorboard==2.16.2"
pip install "tf2onnx==1.16.1" "onnx==1.16.0" "onnxruntime==1.19.2" "protobuf==3.20.3"
pip install "scipy" "scikit-learn" "matplotlib" "joblib" "ipython"

# Optional extras
pip install "netron" "pydot"
sudo apt-get install graphviz   # (Linux) for model_topology.png

3. Verify versions (Python REPL)
-----------------------------------
import tensorflow, tf2onnx, onnx, onnxruntime, google.protobuf
print("TF:", tensorflow.__version__)
print("tf2onnx:", tf2onnx.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("protobuf:", google.protobuf.__version__)

Expected:
TF: 2.16.1
tf2onnx: 1.16.1
onnx: 1.16.0
onnxruntime: 1.19.2
protobuf: 3.20.3

4. Run training
------------------
python3 LSTM_ODE.py

5. View model in Netron
---------------------------
Open https://netron.app
→ drag in  artifacts/model.onnx  (detailed graph)
→ or  artifacts/my_model.keras   (compact graph)

───────────────────────────────────────────────────────────────
NOTES:
- Set USE_DECOMPOSED = True in the script to export a detailed
  gate-level ONNX graph (expanded LSTMCell view).
- Set USE_DECOMPOSED = False for a compact single LSTM node.
───────────────────────────────────────────────────────────────
Summary of commands (for Linux terminal / prompt)
# 1. Create & activate environment
python3 -m venv venv_sussman
source venv_sussman/bin/activate

# 2. Install compatible dependencies
#on windows subsystem for linux, ./bin/pip ....
pip install --upgrade pip
pip install "tensorflow==2.16.1" "tensorboard==2.16.2"
#onnxruntime==1.16.3 for Erik Klassen's old Macbook
pip install "tf2onnx==1.16.1" "onnx==1.16.0" "onnxruntime==1.19.2" "protobuf==3.20.3"
pip install "scipy" "scikit-learn" "matplotlib" "joblib" "ipython" "netron" "pydot"
sudo apt-get install graphviz  # (optional for model diagram)

# 3. Run the script
python3 LSTM_ODE.py
---------------------------------------------------------------
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import joblib

# ------------ Toggle: decomposed vs fused LSTM ------------
USE_DECOMPOSED = True   # True => expanded gate-level graph; False => single fused LSTM

# Optional deps (skip ONNX if not present/compatible)
try:
    import tf2onnx
except Exception as e:
    tf2onnx = None
    print("[INFO] tf2onnx not available:", e)

try:
    import onnx
except Exception as e:
    onnx = None
    print("[INFO] onnx not available:", e)

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    print("[INFO] onnxruntime not available:", e)

try:
    import netron  # not required
except Exception as e:
    netron = None
    print("[INFO] netron (python package) not available:", e)

# -----------------------------
# 1) ODE 
# -----------------------------

#osc_ode is called from solve_ivp
#nodes=0,1,...,n_nodes-1
def osc_ode(t, y, n_nodes,speed):
    dx=1.0/(n_nodes-1.0)
    dxdt=[]
    for i in range(n_nodes):
        local_dxdt=0.0
        if (i==0):
            local_dxdt=-speed*(y[i]-y[n_nodes-2])/dx
        elif (i==n_nodes-1):
            local_dxdt=-speed*(y[i]-y[i-1])/dx
        else:
            local_dxdt=-speed*(y[i]-y[i-1])/dx
        dxdt.append(local_dxdt)

    return dxdt

def make_trajectory(n_nodes=2,speed=1.0,amp=1.0,phase=0.0,T=20.0, dt=0.01):
    n = int(np.floor(T / dt + 1e-12))
    t_eval = np.linspace(0.0, n * dt, n + 1)
    dx=1.0/(n_nodes-1.0)
    y0=[]
    for i in range(n_nodes):
        y0.append(amp*np.sin(2.0*np.pi*i*dx+phase))
    sol = solve_ivp(
        fun=lambda t, y: osc_ode(t, y, n_nodes,speed),
        t_span=(0.0, T),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12)
     # [ [y^{0}] [y^{1}] ... [y^{n}] ]
    Y = np.vstack(sol.y).T
    #print(Y)
    #exit()
    return sol.t, Y


# -----------------------------
# 2) Windowed supervised data
# -----------------------------
def make_supervised_multi(train_scaled, window=50, horizon=1):
    print("make_supervise_multi")
    print("window=")
    print(window)
    print("horizon=")
    print(horizon)
    X, Y = [], []
    Tlen = len(train_scaled)
    print("Tlen")
    print(Tlen)
    # [ [y^{0}] [y^{1}] ... [y^{n}] ]
    #print(train_scaled)
    #print(train_scaled[1,0])
    #TEMP_ARRAY=np.array(train_scaled,np.float32)
    #print(TEMP_ARRAY)
    #print(TEMP_ARRAY[0,0])
    #print(TEMP_ARRAY[1,0])
    #print(TEMP_ARRAY[0,1])
    #exit()
    number_windows=Tlen-window-horizon+1
    print("window=")
    print(window)
    print("number_windows=")
    print(number_windows)
    for i in range(number_windows):
        X.append(train_scaled[i:i+window, :])
        Y.append(train_scaled[i+window:i+window+horizon, :].ravel())
    return np.array(X, np.float32), np.array(Y, np.float32)

# -----------------------------
# 3) Models
# -----------------------------
def build_lstm_fused(window, out_dim, units=96, dropout=0.10, num_dense=64,n_nodes=2):
    """Standard fused LSTM layer (single ONNX LSTM node)."""
    model = keras.Sequential([
        layers.Input(name="input_layer", shape=(window, n_nodes)),
        layers.LSTM(units, name="lstm"),
        layers.Dropout(dropout, name="dropout"),
        layers.Dense(num_dense, activation="relu", name="dense"),
        layers.Dense(out_dim, name="dense_1")
    ], name="sequential")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def build_lstm_decomposed(window, out_dim, units=96, dropout=0.10, num_dense=64,n_nodes=2):
    """
    Decomposed LSTM via RNN(LSTMCell, unroll=True). Exports as primitive ops,
    so Netron shows gate-level graph (MatMul/Add/Sigmoid/Tanh/etc.).
    """
    inputs = layers.Input(name="input_layer", shape=(window, n_nodes))
    x = layers.RNN(layers.LSTMCell(units, name="lstm_cell"),
                   return_sequences=False,  # last state only
                   unroll=True,              # expand time steps
                   name="lstm_unrolled")(inputs)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(num_dense, activation="relu", name="dense")(x)
    outputs = layers.Dense(out_dim, name="dense_1")(x)
    model = keras.Model(inputs, outputs, name="sequential_decomposed")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

# -----------------------------
# 4) Free-roll
# -----------------------------
def freeroll(model, scaler, y_hist, steps, horizon=1,n_nodes=2):
    window = y_hist.shape[0]
    out = []
    sh = scaler.transform(y_hist).astype(np.float32)
    for _ in range(steps):
        y_in = sh.reshape(1, window, n_nodes)
        yhat_scaled = model.predict(y_in, verbose=0)[0]
        yhat = scaler.inverse_transform(yhat_scaled.reshape(horizon, n_nodes))
        out.append(yhat[0])
        sh = np.vstack([sh[1:], yhat_scaled.reshape(horizon, n_nodes)])[-window:]
    return np.array(out)

# -----------------------------
# 5) Save everything
# -----------------------------
def save_all_artifacts(model, history, scaler, Xtr, Ytr, cfg):
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    print(f"\n[SAVE] Writing artifacts to: {out.resolve()}")

    # A) Keras model
    model.save(out / "my_model.keras")

    # A2) Keras 3-safe SavedModel export (optional)
    try:
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, cfg['window'], cfg['n_nodes']), dtype=tf.float32)])
        def serving_fn(x):
            return {"outputs": model(x)}
        tf.saved_model.save(
            obj=model,
            export_dir=str(out / "savedmodel"),
            signatures={"serving_default": serving_fn}
        )
        print("[SAVE] SavedModel exported to artifacts/savedmodel")
    except Exception as e:
        print("[INFO] Skipping SavedModel export:", e)

    # B) Diagram (Graphviz+pydot required)
    try:
        plot_model(model, to_file=str(out / "model_topology.png"),
                   show_shapes=True, show_layer_names=True, dpi=200)
        print("[SAVE] model_topology.png")
    except Exception as e:
        print("[WARN] Could not create model diagram (graphviz/pydot missing?):", e)

    # E) ONNX export/check (optional)
    if tf2onnx is None:
        print("[INFO] Skipping ONNX export (tf2onnx not installed/compatible).")
    else:
        try:
            tspecs = [tf.TensorSpec(i.shape, name=(i.name or f"input_{k}"))
                      for k, i in enumerate(model.inputs)]
            onnx_path = out / "model.onnx"
            tf2onnx.convert.from_keras(model=model,
                                       input_signature=tspecs,
                                       opset=18,
                                       output_path=str(onnx_path))
            print("[SAVE] Exported ONNX:", onnx_path)

            if onnx is not None:
                onnx.checker.check_model(onnx.load(str(onnx_path)))
                print("[CHECK] ONNX model check passed.")
            if ort is not None:
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                print("[CHECK] ONNX Runtime session OK. Inputs:", [i.name for i in sess.get_inputs()])
            else:
                print("[INFO] onnxruntime not installed; skipping quick inference.")
        except Exception as e:
            print("[WARN] ONNX export or check failed:", e)

    # F) Netron guidance
    print("\n[NETRON] Open https://netron.app and load either:")
    print(f"         - {out / 'model.onnx'} (if present)")
    print(f"         - {out / 'my_model.keras'}")

# -----------------------------
# 6) Main
# -----------------------------
if __name__ == "__main__":
    tf.random.set_seed(42); np.random.seed(42)

    # Config
    n_nodes=11 # 0...n_nodes-1
    dx=1.0/(n_nodes-1.0)
    speed=1.0
    dt=0.5*dx/speed
    T=0.5
    #window, horizon = 50, 1
    window, horizon = 4, 1
    #EPOCHS, BATCH = 10, 256
    EPOCHS, BATCH = 1000, 256
    #UNITS, DROPOUT, NUM_DENSE = 96, 0.10, 64
    UNITS, DROPOUT, NUM_DENSE = 8, 0.10, 8

    print(f"[CFG] T={T}, dt={dt}, window={window}, horizon={horizon}, "
          f"epochs={EPOCHS}, units={UNITS}, decomposed={USE_DECOMPOSED}")

    # Data
    amp=2.0
    phase=np.pi
     # [ [y^{0}] [y^{1}] ... [y^{n}] ]
    all_t, all_Y = make_trajectory(n_nodes=n_nodes,speed=speed,amp=amp,phase=phase,T=T, dt=dt)

    # 1. make sure mean with respect to time is zero for each space location.
    #    Save XMEAN_i   i=0,...,n_nodes-1
    # 2. X=U SIGMA WSTAR
    # 3. TSVD=X W

    n_nodes_reduced=n_nodes

    scaler = MinMaxScaler()
    scaler.fit(np.vstack(all_Y))
    train_scaled = scaler.transform(all_Y)

    print("n_nodes")
    print(n_nodes)
    print("n_nodes_reduced")
    print(n_nodes_reduced)
    print("make_supervised_multi (Xtr,Ytr)")
    Xtr, Ytr = make_supervised_multi(train_scaled, window=window, horizon=horizon)
    print(f"[DATA] Xtr: {Xtr.shape}, Ytr: {Ytr.shape} ")

    # Model + train
    if USE_DECOMPOSED:
        model = build_lstm_decomposed(window=window, out_dim=n_nodes_reduced*horizon, units=UNITS, dropout=DROPOUT,num_dense=NUM_DENSE,n_nodes=n_nodes_reduced)
    else:
        model = build_lstm_fused(window=window, out_dim=n_nodes_reduced*horizon, units=UNITS, dropout=DROPOUT,num_dense=NUM_DENSE,n_nodes=n_nodes_reduced)

    model.summary()
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        Xtr, Ytr,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=1
    )

    # Save everything
    save_all_artifacts(
        model=model,
        history=history,
        scaler=scaler,
        Xtr=Xtr, Ytr=Ytr,
        cfg=dict(T=T, dt=dt, window=window, horizon=horizon,n_nodes=n_nodes_reduced,units=UNITS, dropout=DROPOUT)
    )

    print("\n[EVAL] Free-roll on an extended time …")
    print("amp,phase: ")
    print(amp,phase)
    T_free_roll=12.0*dt
    #T_free_roll=T
    #T_free_roll=5.0*dt
    print("T,T_free_roll: ")
    print(T,T_free_roll)

    #expected solution
    t_te, Y_te = make_trajectory(n_nodes=n_nodes,speed=speed,amp=amp,phase=phase,T=T_free_roll, dt=dt)

    # 1. adjust each node location by XMEAN_i   i=0,...,n_nodes-1
    # 2. TSVD=X W

    init_hist = Y_te[:window, :]
    steps     = len(Y_te) - window

    Y_pred    = freeroll(model, scaler, init_hist, steps=steps, horizon=horizon,n_nodes=n_nodes_reduced)
    Y_pred_full = np.vstack([init_hist, Y_pred])

    # for Y_te and Y_pred
    # 1. X=TSVD * W^{star}
    # 2. restore the XMEAN_i operation i=0,....,n_nodes-1

    y_true = Y_te[window:, :]
    y_hat  = Y_pred_full[window:, :]

    for i in range(n_nodes):
        rmse=np.sqrt(mean_squared_error(y_true[:,i],y_hat[:,i]))
        r2 = r2_score(y_true[:, i], y_hat[:, i])
        print("i,rmse,r2")
        print(i,rmse,r2)

    out = Path("artifacts")
    plt.figure(figsize=(11, 5))
    plt.subplot(1,2,1)
    plt.plot(t_te, Y_te[:,0], label="True x(t,0)")
    plt.plot(t_te, Y_pred_full[:,0], "--", label="LSTM x̂(t)")
    plt.axvline(t_te[window], color="k", ls=":", lw=1)
    plt.title("x(t,0)")
    plt.xlabel("Time [s]"); plt.ylabel("x(t,0)"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_te, Y_te[:,1], label="True x(t,1)")
    plt.plot(t_te, Y_pred_full[:,1], "--", label="LSTM v̂(t)")
    plt.axvline(t_te[window], color="k", ls=":", lw=1)
    plt.title("x(t,1)")
    plt.xlabel("Time [s]"); plt.ylabel("x(t,1)"); plt.legend()
    plt.tight_layout()
    plt.savefig(out / "timeseries_eval.png", dpi=200)
    plt.show()

    print("\n[DONE] Artifacts saved in:", out.resolve())
    print("      Open https://netron.app and load artifacts/model.onnx (if exported) or my_model.keras.")
