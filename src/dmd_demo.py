import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 1. DATASETS

def make_linear_oscillator_data(nt=120, dt=0.05, alpha=0.00, omega=1.50, x0=np.array([1.0, 0.0])):
    A_true = np.array([
        [np.exp(alpha*dt)*np.cos(omega*dt), -np.exp(alpha*dt)*np.sin(omega*dt)],
        [np.exp(alpha*dt)*np.sin(omega*dt),  np.exp(alpha*dt)*np.cos(omega*dt)]
    ])

    X = np.zeros((nt, 2), dtype=float)
    X[0] = x0
    for k in range(nt - 1):
        X[k+1] = A_true @ X[k]

    lam_true = np.linalg.eigvals(A_true)
    t = np.arange(nt) * dt
    return X, t, A_true, lam_true

def make_spatiotemporal_data(nx=128, nt=140, noise_level=0.0):
    x = np.linspace(-6, 6, nx)
    t = np.linspace(0, 6*np.pi, nt)
    xgrid, tgrid = np.meshgrid(x, t)
    f1 = (1.0 / np.cosh(xgrid + 2.5)) * np.cos(2.2 * tgrid)
    f2 = (1.6 / np.cosh(xgrid - 0.5)) * np.tanh(xgrid - 0.5) * np.sin(2.8 * tgrid)
    X = f1 + f2
    if noise_level > 0:
        X = X + noise_level * np.random.randn(*X.shape)
    return X, x, t
# 2. 

def build_snapshot_matrices(X):
    Xsnap = X.T
    X1 = Xsnap[:, :-1]
    X2 = Xsnap[:, 1:]
    return X1, X2

def rel_fro_error(X_true, X_pred):
    return np.linalg.norm(X_true - X_pred, ord='fro') / np.linalg.norm(X_true, ord='fro')

def train_test_split_snapshots(X, train_ratio=0.8):
    nt = X.shape[0]
    ntrain = int(train_ratio * nt)
    return X[:ntrain], X[ntrain:]

def exact_dmd(X, r):
    X1, X2 = build_snapshot_matrices(X)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(r, len(S))
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh[:r, :].T
    Ar = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    lam, W = np.linalg.eig(Ar)
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
    x1 = X1[:, 0]
    b = np.linalg.pinv(Phi) @ x1
    return {
        "X1": X1,
        "X2": X2,
        "U": U, "S": S, "Vh": Vh,
        "Ur": Ur, "Sr": Sr, "Vr": Vr,
        "Ar": Ar,
        "lam": lam,
        "W": W,
        "Phi": Phi,
        "b": b,
        "r": r
    }

def reconstruct_from_dmd(model, n_steps):
    Phi = model["Phi"]
    lam = model["lam"]
    b = model["b"]
    n_features = Phi.shape[0]
    Xrec = np.zeros((n_features, n_steps), dtype=complex)
    for k in range(n_steps):
        Xrec[:, k] = Phi @ (b * (lam ** k))
    return Xrec.T 
# 4. TIME-DELAY  DMD

def make_delay_embedding(X, d):
    nt, nx = X.shape
    Y = []
    for k in range(nt - d + 1):
        Y.append(X[k:k+d].reshape(-1))
    return np.array(Y)

def delay_dmd(X, r, d=2):
    Y = make_delay_embedding(X, d)
    model = exact_dmd(Y, r)
    model["delay"] = d
    model["embedded_data"] = Y
    return model

def reconstruct_from_delay_dmd(model, n_steps, nx):
    Yrec = reconstruct_from_dmd(model, n_steps)  
    Xrec = Yrec[:, :nx]
    return Xrec
# 5. Evalutions

def evaluate_exact_dmd(X, r, train_ratio=0.8):
    X_train, X_test = train_test_split_snapshots(X, train_ratio=train_ratio)
    model = exact_dmd(X_train, r=r)
    Xrec_train = reconstruct_from_dmd(model, n_steps=len(X_train)).real
    rec_err = rel_fro_error(X_train, Xrec_train)
    Xpred_all = reconstruct_from_dmd(model, n_steps=len(X_train) + len(X_test)).real
    Xpred_test = Xpred_all[len(X_train):]
    pred_err = rel_fro_error(X_test, Xpred_test)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "Xrec_train": Xrec_train,
        "Xpred_test": Xpred_test,
        "rec_err": rec_err,
        "pred_err": pred_err
    }

def evaluate_delay_dmd(X, r, d=2, train_ratio=0.8):
    X_train, X_test = train_test_split_snapshots(X, train_ratio=train_ratio)
    nx = X.shape[1]
    model = delay_dmd(X_train, r=r, d=d)
    n_train_emb = len(X_train) - d + 1
    Xrec_train = reconstruct_from_delay_dmd(model, n_steps=n_train_emb, nx=nx).real
    Xtrain_target = X_train[:n_train_emb]
    rec_err = rel_fro_error(Xtrain_target, Xrec_train)
    total_steps = n_train_emb + len(X_test)
    Xpred_all = reconstruct_from_delay_dmd(model, n_steps=total_steps, nx=nx).real
    Xpred_test = Xpred_all[n_train_emb:]
    pred_err = rel_fro_error(X_test[:len(Xpred_test)], Xpred_test)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "Xrec_train": Xrec_train,
        "Xtrain_target": Xtrain_target,
        "Xpred_test": Xpred_test,
        "rec_err": rec_err,
        "pred_err": pred_err
    }


def rank_sweep_exact(X, ranks, train_ratio=0.8):
    rec_errs, pred_errs = [], []
    for r in ranks:
        out = evaluate_exact_dmd(X, r=r, train_ratio=train_ratio)
        rec_errs.append(out["rec_err"])
        pred_errs.append(out["pred_err"])
    return np.array(rec_errs), np.array(pred_errs)


def rank_sweep_delay(X, ranks, d=2, train_ratio=0.8):
    rec_errs, pred_errs = [], []
    for r in ranks:
        out = evaluate_delay_dmd(X, r=r, d=d, train_ratio=train_ratio)
        rec_errs.append(out["rec_err"])
        pred_errs.append(out["pred_err"])
    return np.array(rec_errs), np.array(pred_errs)
# 6. PLOTTING

def plot_singular_values(model, title="Singular Values"):
    S = model["S"]
    plt.figure(figsize=(5, 3))
    plt.semilogy(S, 'o-')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.tight_layout()
    plt.show()

def plot_eigs(lam, title="DMD Eigenvalues"):
    theta = np.linspace(0, 2*np.pi, 400)
    plt.figure(figsize=(4, 4))
    plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
    plt.scatter(lam.real, lam.imag, c='r')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.tight_layout()
    plt.show()

def plot_timeseries_2d(X_true, X_pred=None, title="2D trajectory"):
    plt.figure(figsize=(4, 4))
    plt.plot(X_true[:, 0], X_true[:, 1], label="true")
    if X_pred is not None:
        plt.plot(X_pred[:, 0], X_pred[:, 1], '--', label="pred")
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spatiotemporal(X, x, t, title="Field"):
    xgrid, tgrid = np.meshgrid(x, t)
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(xgrid, tgrid, X.real, shading='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    plt.show()

def plot_reconstruction_comparison(X_true, X_rec, x, t, title_prefix="Exact DMD"):
    xgrid, tgrid = np.meshgrid(x, t[:len(X_rec)])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    axs[0].pcolormesh(xgrid, tgrid, X_true[:len(X_rec)].real, shading='auto')
    axs[0].set_title("Original")
    axs[1].pcolormesh(xgrid, tgrid, X_rec.real, shading='auto')
    axs[1].set_title("Reconstruction")
    axs[2].pcolormesh(xgrid, tgrid, (X_true[:len(X_rec)] - X_rec).real, shading='auto')
    axs[2].set_title("Residual")
    fig.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

def plot_rank_errors(ranks, rec_err_exact, pred_err_exact, rec_err_delay, pred_err_delay):
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, rec_err_exact, 'o-', label='exact rec')
    plt.plot(ranks, pred_err_exact, 'o--', label='exact pred')
    plt.plot(ranks, rec_err_delay, 's-', label='delay rec')
    plt.plot(ranks, pred_err_delay, 's--', label='delay pred')
    plt.xlabel("rank r")
    plt.ylabel("relative error")
    plt.title("Effect of rank truncation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# 7. RUN THE PROJECT

X_lin, t_lin, A_true, lam_true = make_linear_oscillator_data(nt=150, dt=0.05, alpha=-0.01, omega=1.8)
out_lin = evaluate_exact_dmd(X_lin, r=2, train_ratio=0.8)

print("True eigenvalues:", lam_true)
print("Exact DMD eigenvalues:", out_lin["model"]["lam"])
print("Linear reconstruction error:", out_lin["rec_err"])
print("Linear prediction error:", out_lin["pred_err"])

plot_singular_values(out_lin["model"], title="Singular Values (Linear Dataset)")
plot_eigs(out_lin["model"]["lam"], title="Exact DMD Eigenvalues (Linear Dataset)")
plot_timeseries_2d(out_lin["X_test"], out_lin["Xpred_test"], title="Linear oscillator prediction")

X_sp, x_sp, t_sp = make_spatiotemporal_data(nx=128, nt=150, noise_level=0.01)

out_ex = evaluate_exact_dmd(X_sp, r=4, train_ratio=0.8)
out_td = evaluate_delay_dmd(X_sp, r=4, d=3, train_ratio=0.8)

print("Spatio-temporal exact rec error:", out_ex["rec_err"])
print("Spatio-temporal exact pred error:", out_ex["pred_err"])
print("Spatio-temporal delay rec error:", out_td["rec_err"])
print("Spatio-temporal delay pred error:", out_td["pred_err"])

plot_spatiotemporal(X_sp, x_sp, t_sp, title="Raw spatio-temporal data")
plot_singular_values(out_ex["model"], title="Singular Values (Main Dataset)")
plot_eigs(out_ex["model"]["lam"], title="Exact DMD Eigenvalues (Main Dataset)")
plot_reconstruction_comparison(out_ex["X_train"], out_ex["Xrec_train"], x_sp, t_sp[:len(out_ex["X_train"])], title_prefix="Exact DMD")
plot_reconstruction_comparison(out_td["Xtrain_target"], out_td["Xrec_train"], x_sp, t_sp[:len(out_td["Xtrain_target"])], title_prefix="Time-delay DMD")

ranks = [2, 4, 6, 8, 10]
rec_e, pred_e = rank_sweep_exact(X_sp, ranks, train_ratio=0.8)
rec_d, pred_d = rank_sweep_delay(X_sp, ranks, d=3, train_ratio=0.8)
plot_rank_errors(ranks, rec_e, pred_e, rec_d, pred_d)

r_report = 4
d_report = 3

table_errors = pd.DataFrame({
    "Method": ["Exact DMD", "Time-delay DMD"],
    "Rank r": [r_report, r_report],
    "Delay d": ["-", d_report],
    "Reconstruction Error": [out_ex["rec_err"], out_td["rec_err"]],
    "Prediction Error": [out_ex["pred_err"], out_td["pred_err"]]
})

table_errors_rounded = table_errors.copy()
table_errors_rounded["Reconstruction Error"] = table_errors_rounded["Reconstruction Error"].map(lambda x: f"{x:.4f}")
table_errors_rounded["Prediction Error"] = table_errors_rounded["Prediction Error"].map(lambda x: f"{x:.4f}")

print(table_errors_rounded)
table_errors_rounded.to_csv("Table1_relative_errors.csv", index=False)
