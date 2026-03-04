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
# 1) ODE & trajectories
# -----------------------------
@dataclass
class ODEParams:
    omega0: float = 2.0 * np.pi * 1.0
    zeta:   float = 0.05

# x(0)=x0
# v(0)=v0
# omega0=2 pi
# zeta=1/20
#
# dx/dt=v
# dv/dt=-2 zeta omega0 v(t) - omega0^2 x(t)
#
# algorithm to generate the data:
# 0<= t <= TSTOP  e.g. dy/dt=-y(t)  solution is y(t)=e^{-t}
# t0,...,tN  
# t0=0.0
# t1=dt
# t2=2 * dt
#  ....
# TSTOP=N * dt
#
# TSTOP, DT, N * DT = TSTOP
# x0,x1,x2,x3, ..., xN
# v0,v1,v2,v3, ..., vN
#
# training data:
# a) the number of trajectories are "n_traj"
# b) 0.8 * n_traj is the number of trajectories to train the network.
# c) 0.2 * n_traj is the number of trajectories to test the network.
# total amount of data to train the network is:
# 0.8*n_traj*N*(2)  the 2 is because we have x and v.
#
# 1. train the network
# 2. the network make a prediction for xN+1, xN+2, ....,xN+P,vN+1,vN+2,...
#    vN+P
#    compare: error=sqrt( (1/P)*sum_{i=N+1}^{N+P} (x_{i}-x_{i}^{exact})^{2}+
#     (v_{i}-v_{i}^{exact})^{2})
def osc_ode(t, y, p: ODEParams):
    x, v = y
    dxdt = v
    dvdt = -2.0 * p.zeta * p.omega0 * v - (p.omega0 ** 2) * x
    return [dxdt, dvdt]

def make_trajectory(p: ODEParams, x0=1.0, v0=0.0, T=20.0, dt=0.01):
    n = int(np.floor(T / dt + 1e-12))
    t_eval = np.linspace(0.0, n * dt, n + 1)
    sol = solve_ivp(
        fun=lambda t, y: osc_ode(t, y, p),
        t_span=(0.0, T),
        y0=[x0, v0],
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    Y = np.vstack((sol.y[0], sol.y[1])).T
    return sol.t, Y

def generate_dataset(n_traj=24, T=20.0, dt=0.01, seed=123):
    print("\n[DATA] Generating synthetic trajectories …")
    print("       n_traj, T, dt, nsteps =", n_traj, T, dt, T/dt)
    rng = np.random.default_rng(seed)
    p = ODEParams()
    all_t, all_Y = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(-2.0, 2.0)
        v0 = rng.uniform(-2.0, 2.0)
        t, Y = make_trajectory(p, x0=x0, v0=v0, T=T, dt=dt)
        all_t.append(t); all_Y.append(Y)
    print("       generated", len(all_Y), "trajectories.")
    print("len(all_t) ")
    print(len(all_t)) #should be equal to the number of trajectories
    print("len(all_Y) ")
    print(len(all_Y)) #should be equal to the number of trajectories
    return all_t, all_Y, p

# -----------------------------
# 2) Windowed supervised data
# -----------------------------
def make_supervised_multi(trajs, window=50, horizon=1):
    print("make_supervise_multi")
    print("window=")
    print(window)
    print("horizon=")
    print(horizon)
    X, Y = [], []
    i_traj=0
    for Ytraj in trajs:
        Tlen = len(Ytraj)
        print("i_traj,Tlen")
        print(i_traj,Tlen)
        number_windows=Tlen-window-horizon+1
        print("window=")
        print(window)
        print("number_windows=")
        print(number_windows)
        for i in range(number_windows):
            X.append(Ytraj[i:i+window, :])
            #numpy.ravel(a, order='C')[source]
            #Return a contiguous flattened array.
            #A 1-D array, containing the elements of the input, 
            #is returned. A copy is made only if needed.
            #As of NumPy 1.10, the returned array will have the 
            #same type as the input array. (for example, a 
            #masked array will be returned for a masked array input)
            Y.append(Ytraj[i+window:i+window+horizon, :].ravel())
        i_traj=i_traj+1
    return np.array(X, np.float32), np.array(Y, np.float32)

# -----------------------------
# 3) Models
# -----------------------------

#out_dim=2*horizon for the ODE problem.
def build_lstm_decomposed(window, out_dim, units=96, dropout=0.10, num_dense=64):
    """
    Decomposed LSTM via RNN(LSTMCell, unroll=True). Exports as primitive ops,
    so Netron shows gate-level graph (MatMul/Add/Sigmoid/Tanh/etc.).
    """
    inputs = layers.Input(name="input_layer", shape=(window, 2))
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
def freeroll(model, scaler, y_hist, steps, horizon=1):
    window = y_hist.shape[0]
    out = []
    sh = scaler.transform(y_hist).astype(np.float32)
    for _ in range(steps):
        y_in = sh.reshape(1, window, 2)
        yhat_scaled = model.predict(y_in, verbose=0)[0]
        yhat = scaler.inverse_transform(yhat_scaled.reshape(horizon, 2))
        out.append(yhat[0])
        sh = np.vstack([sh[1:], yhat_scaled.reshape(horizon, 2)])[-window:]
    return np.array(out)

# -----------------------------
# 5) Save everything
# -----------------------------
def save_all_artifacts(model, history, scaler, params, Xtr, Ytr, cfg):
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    print(f"\n[SAVE] Writing artifacts to: {out.resolve()}")

    # A) Keras model
    model.save(out / "my_model.keras")

    # A2) Keras 3-safe SavedModel export (optional)
    try:
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, cfg['window'], 2), dtype=tf.float32)])
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

    # C) Training history
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    np.savetxt(out / "history.csv",
               np.array([hist.get("loss", []), hist.get("val_loss", [])]).T,
               delimiter=",", header="loss,val_loss", comments="")
    with open(out / "history.json", "w") as f:
        json.dump(hist, f, indent=2)
    plt.figure()
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "history.png", dpi=200); plt.close()

    # D) Scaler + config
    joblib.dump(scaler, out / "minmax_scaler.joblib")
    config = dict(**cfg,
                  omega0=float(params.omega0),
                  zeta=float(params.zeta),
                  Xtr_shape=tuple(Xtr.shape),
                  Ytr_shape=tuple(Ytr.shape))
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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

#number of equations in the system is 2,
#the input layer shape is (None,window,2) None=>Batch size is initialized 
#later (model.fit).
#lstm_unrolled shape=(None, UNITS) number of 
#parameters lstm_unrolled=4*(UNITS*2+UNITS*UNITS+UNITS)
#UNITS=number neurons in the LSTM layer.
#f_t=sigma(Wf x_t + Uf h_{t-1} +bf)
#i_t=sigma(Wi x_t + Ui h_{t-1} +bi)
#o_t=sigma(Wo x_t + Uo h_{t-1} +bo)
#\tilde{c}_t=sigma(Wc x_t + Uc h_{t-1} +bc)
#4*(UNITS*2+UNITS*UNITS+UNITS)
#referring to the wikipedia LSTM website for "LSTM with a forget gas:
#d=2
#h=UNITS
#dense=(None,NUM_DENSE) 
#number of parameters(dense)=UNITS*NUM_DENSE (weight)+NUM_DENSE (bias)
#dense_1=(None,2) number parameters(dense_1)=NUM_DENSE * 2 + 2  weight matrix is
#NUM_DENSE x 2  bias matrix is 2 x 1
#loss function="mse"="mean square error"
#loss=sum_k=1^{num_windows*num_traj} sum_j=1^2 
#   ||x^{expect}_{j,k}-x^{net}(x_{j,k},theta)||^2
#The smaller the size of "window," the more vectorizable the code is,
#but less accurate!
#The larger window size is less vectorizable, but more accurate.
    # Config
    #T, dt = 20.0, 0.01
    T, dt = 1.0, 0.01  #t0=0 t1=0.01 t2=0.02 .... t100=1
    #T, dt = 0.08, 0.01
    #window, horizon = 50, 1
    window, horizon = 50, 1  #window1=t0 ...t49  window2=t1...t50 ...
    #window51=t50 ... t99
    # total number of windows: n_traj * (51)  51=number of windows for each
    # trajectory.
    #n_traj, EPOCHS, BATCH = 24, 10, 256
    n_traj, EPOCHS, BATCH = 24, 100, 256
    #n_traj, EPOCHS, BATCH = 4, 2, 256
    #UNITS, DROPOUT, NUM_DENSE = 96, 0.10, 64
    UNITS, DROPOUT, NUM_DENSE = 128, 0.10, 96
    #UNITS, DROPOUT, NUM_DENSE = 6, 0.10, 4

    print(f"[CFG] T={T}, dt={dt}, window={window}, horizon={horizon}, "
          f"n_traj={n_traj}, epochs={EPOCHS}, units={UNITS}")

    # in the poster:
    # T=1.0
    # dt=0.01
    # window=50
    # number of windows=(T/dt - window)
    # n_traj=24
    # total number of windows=n_traj * numbr of windows
    # training windows=0.8 * total numbr of windows
    # Data
    all_t, all_Y, params = generate_dataset(n_traj=n_traj, T=T, dt=dt, seed=123)
    # total number of windows=n_traj * ((T/dt)-window)
    # total number of training windows=0.8 * total_number_of_windows.
    # total number of test windows=0.2 * total_number_of_windows.
    n_train = int(0.8 * len(all_Y))
    print("len(all_Y)")
    print(len(all_Y))
    print("n_train")
    print(n_train)

    train_trajs, val_trajs = all_Y[:n_train], all_Y[n_train:]

    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_trajs))
    train_scaled = [scaler.transform(y) for y in train_trajs]
    val_scaled   = [scaler.transform(y) for y in val_trajs]

    print("n_traj")
    print(n_traj)
     #training data
    print("make_supervised_multi (Xtr,Ytr)")
    Xtr, Ytr = make_supervised_multi(train_scaled, window=window, horizon=horizon)
     #validation data
    print("make_supervised_multi (Xva,Yva)")
    Xva, Yva = make_supervised_multi(val_scaled,   window=window, horizon=horizon)
    print(f"[DATA] Xtr: {Xtr.shape}, Ytr: {Ytr.shape} | Xva: {Xva.shape}, Yva: {Yva.shape}")

    # Model + train
    model = build_lstm_decomposed(window=window, out_dim=2*horizon, units=UNITS, dropout=DROPOUT,num_dense=NUM_DENSE)

    model.summary()
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=1
    )

    # Save everything
    # to be analyzed by ONNX
    save_all_artifacts(
        model=model,
        history=history,
        scaler=scaler,
        params=params,
        Xtr=Xtr, Ytr=Ytr,
        cfg=dict(T=T, dt=dt, window=window, horizon=horizon, n_traj=n_traj, units=UNITS, dropout=DROPOUT)
    )

    # Evaluate on a fresh trajectory + plots
    # The following "test" data will be compared to the LSTM predicted data.
    # Y_pred should correspond to Y_te for the first 80 percent of the data.
    print("\n[EVAL] Free-roll on a fresh test trajectory …")
    t_te, Y_te = make_trajectory(params, x0=1.25, v0=-0.8, T=T, dt=dt)
    init_hist = Y_te[:window, :]
    steps     = len(Y_te) - window
    Y_pred    = freeroll(model, scaler, init_hist, steps=steps, horizon=horizon)
    Y_pred_full = np.vstack([init_hist, Y_pred])

    y_true = Y_te[window:, :]
    y_hat  = Y_pred_full[window:, :]
    rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_hat[:, 0]))
    rmse_v = np.sqrt(mean_squared_error(y_true[:, 1], y_hat[:, 1]))
    r2_x   = r2_score(y_true[:, 0], y_hat[:, 0])
    r2_v   = r2_score(y_true[:, 1], y_hat[:, 1])
    print(f"[EVAL] RMSE_x={rmse_x:.4e}, R2_x={r2_x:.4f} | RMSE_v={rmse_v:.4e}, R2_v={r2_v:.4f}")

    out = Path("artifacts")
    plt.figure(figsize=(11, 5))
    plt.subplot(1,2,1)
    plt.plot(t_te, Y_te[:,0], label="True x(t)")
    plt.plot(t_te, Y_pred_full[:,0], "--", label="LSTM x̂(t)")
    plt.axvline(t_te[window], color="k", ls=":", lw=1)
    plt.title(f"x(t): RMSE={rmse_x:.3e}, R²={r2_x:.3f}")
    plt.xlabel("Time [s]"); plt.ylabel("x"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_te, Y_te[:,1], label="True v(t)")
    plt.plot(t_te, Y_pred_full[:,1], "--", label="LSTM v̂(t)")
    plt.axvline(t_te[window], color="k", ls=":", lw=1)
    plt.title(f"v(t): RMSE={rmse_v:.3e}, R²={r2_v:.3f}")
    plt.xlabel("Time [s]"); plt.ylabel("v"); plt.legend()
    plt.tight_layout()
    plt.savefig(out / "timeseries_eval.png", dpi=200)
    plt.show()

    plt.figure(figsize=(5.5, 5))
    plt.plot(Y_te[:,0], Y_te[:,1], label="True")
    plt.plot(Y_pred_full[:,0], Y_pred_full[:,1], "--", label="LSTM")
    plt.xlabel("x"); plt.ylabel("v")
    plt.title("Phase portrait")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "phase_portrait.png", dpi=200)
    plt.show()

    np.savez(out / "eval_arrays.npz",
             t_te=t_te, Y_te=Y_te, Y_pred_full=Y_pred_full,
             rmse_x=np.array([rmse_x]), rmse_v=np.array([rmse_v]),
             r2_x=np.array([r2_x]), r2_v=np.array([r2_v]))

    print("\n[DONE] Artifacts saved in:", out.resolve())
    print("      Open https://netron.app and load artifacts/model.onnx (if exported) or my_model.keras.")
