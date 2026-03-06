import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from matplotlib.ticker import LogLocator, NullFormatter


# 1) Experimental data (TIME IN HOURS)

VARS_ORDER = ["x1", "x2", "x3", "x4", "x5"]
PARAM_NAMES = ["k12", "k13", "k14", "k15", "k23", "k24", "k25", "k34", "k35", "k45"]

ultrasound = {
    "t": np.array([
        0, 0.0028, 0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.25, 0.3333, 0.4167,
        0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167, 1, 1.0833, 1.1667, 1.25,
        1.3333, 1.4167, 1.5, 1.5833, 1.6667, 1.75, 1.8333, 1.9167, 2.0
    ]),
    "x5": np.array([1.4, 2.02, 2.92, 4.37, 5.37, 9.94, 11.22, 16.6, 22.22, 24.97,
                    23.61, 27.63, 29.58, 31.8, 32.4, 33.02, 33.86, 34.43, 34.93,
                    35.42, 35.91, 36.28, 36.7, 37.03, 37.55, 37.94, 38.25, 38.62, 38.93]),
    "x4": np.array([2.49, 4.1, 5.91, 8.21, 9.55, 16.33, 17.8, 26.53, 34.04, 36.91,
                    33.66, 38.64, 40.43, 42.84, 42.97, 43.01, 43.01, 43.08, 43.06,
                    43.07, 42.99, 42.94, 42.85, 42.67, 42.61, 42.5, 42.39, 42.26, 42.12]),
    "x3": np.array([6.22, 7.07, 7.2, 8.54, 8.94, 14.64, 15.16, 22.44, 26.65, 26.92,
                    23.72, 25.27, 25.12, 25.03, 24.39, 23.8, 23.04, 22.4, 21.98,
                    21.51, 21.1, 20.78, 20.45, 20.24, 19.84, 19.56, 19.36, 19.12, 18.95]),
    "x2": np.array([9.83, 4.96, 2.85, 2.59, 2.31, 2.68, 2.5, 2.65, 2.15, 1.9, 1.6,
                    1.17, 0.8, 0.33, 0.24, 0.17, 0.09, 0.09, 0.03, 0, 0, 0, 0,
                    0.06, 0, 0, 0, 0, 0]),
    "x1": np.array([80.06, 81.85, 81.12, 76.29, 73.83, 56.41, 53.32, 31.78, 14.94,
                    9.3, 17.41, 7.29, 4.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0]),
}

ball_mill = {
    "t": np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3.5, 4.5, 8.5]),
    "x5": np.array([10.11, 50.24, 62.58, 69.37, 72.98, 76.22, 80.94, 86.4, 90.24, 94.49, 97.79, 99.96]),
    "x4": np.array([9.83, 20.87, 18.07, 16.81, 14.94, 13.36, 11.72, 9.39, 7.07, 3.84, 1.99, 0.04]),
    "x3": np.array([7.01, 20.35, 14.36, 11.17, 7.99, 7.94, 5.78, 3.47, 2.15, 1.01, 0.22, 0.0]),
    "x2": np.array([0.0, 8.4, 4.91, 2.61, 2.1, 2.24, 1.54, 0.73, 0.54, 0.66, 0.0, 0.0]),
    "x1": np.array([73.05, 0.14, 0.08, 0.04, 1.99, 0.24, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]),
}


# 2) Kinetic model 

def system(t, y, k):
    k12, k13, k14, k15, k23, k24, k25, k34, k35, k45 = k
    x1, x2, x3, x4, x5 = y

    dx1 = -(k12 + k13 + k14 + k15) * x1
    dx2 = (k12 * x1) - (k23 + k24 + k25) * x2
    dx3 = (k13 * x1) + (k23 * x2) - (k34 + k35) * x3
    dx4 = (k14 * x1) + (k24 * x2) + (k34 * x3) - (k45 * x4)
    dx5 = (k15 * x1) + (k25 * x2) + (k35 * x3) + (k45 * x4)

    return [dx1, dx2, dx3, dx4, dx5]


def simulate(t_eval, y0, k):
    sol = solve_ivp(
        fun=lambda t, y: system(t, y, k),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failure: {sol.message}")
    return sol.y.T


# 3) Parameter fitting

def fit_model(name, data):
    t = np.asarray(data["t"], dtype=float)
    Yexp = np.vstack([np.asarray(data[v], dtype=float) for v in VARS_ORDER]).T
    y0 = Yexp[0].copy()

    scale = np.maximum(np.max(Yexp, axis=0), 1.0)

    def residuals(k):
        Yfit = simulate(t, y0, k)
        return ((Yfit - Yexp) / scale).ravel()

    k0 = np.full(10, 1.0)
    lb = np.zeros(10)
    ub = np.full(10, np.inf)

    res = least_squares(
        residuals,
        x0=k0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=8000,
    )

    kopt = res.x
    Yfit = simulate(t, y0, kopt)

    sse = float(np.sum((Yfit - Yexp) ** 2))
    rmse = float(np.sqrt(np.mean((Yfit - Yexp) ** 2)))
    y_mean = float(np.mean(Yexp))
    sst = float(np.sum((Yexp - y_mean) ** 2))
    r2 = float(1 - sse / sst) if sst > 0 else float("nan")

    sum_exp = np.sum(Yexp, axis=1)
    sum_fit = np.sum(Yfit, axis=1)

    df = pd.DataFrame({"t_h": t, "t_min": t * 60.0})
    for i, v in enumerate(VARS_ORDER):
        df[f"{v}_experimental"] = Yexp[:, i]
        df[f"{v}_model"] = Yfit[:, i]
    df["sum_experimental"] = sum_exp
    df["sum_model"] = sum_fit
    df.to_csv(f"fit_{name}.csv", index=False)

    with open(f"params_{name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Optimized parameters (h^-1) - {name}\n")
        for n, val in zip(PARAM_NAMES, kopt):
            f.write(f"{n} = {val:.6g}\n")
        f.write("\nFit metrics:\n")
        f.write(f"SSE  = {sse:.6g}\n")
        f.write(f"RMSE = {rmse:.6g}\n")
        f.write(f"R2   = {r2:.6g}\n")
        f.write("\nConservation check (sum of fractions):\n")
        f.write(f"sum_exp: min={float(np.min(sum_exp)):.3f}, max={float(np.max(sum_exp)):.3f}\n")
        f.write(f"sum_fit: min={float(np.min(sum_fit)):.3f}, max={float(np.max(sum_fit)):.3f}\n")

    k_out_x1 = float(np.sum(kopt[0:4]))
    t_half_x1 = float(np.log(2) / k_out_x1) if k_out_x1 > 0 else float("inf")

    return {
        "name": name,
        "t_h": t,
        "t_min": t * 60.0,
        "Yexp": Yexp,
        "Yfit": Yfit,
        "k": kopt,
        "metrics": {"sse": sse, "rmse": rmse, "r2": r2},
        "k_out_x1": k_out_x1,
        "t_half_x1_h": t_half_x1,
        "t_half_x1_min": t_half_x1 * 60.0,
    }


# 4) Style and Figures

def set_mpl_article_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "axes.linewidth": 2.2,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
    })


def _log_safe_time(t_min: np.ndarray) -> np.ndarray:
    t_min = np.asarray(t_min, dtype=float).copy()
    pos = t_min[t_min > 0]
    if pos.size == 0:
        return np.full_like(t_min, 1e-3)
    eps = max(np.min(pos) / 10.0, 1e-3)
    t_min[t_min <= 0] = eps
    return t_min


def apply_mastersizer_grid(ax):
    ax.set_xscale("log")

    major = LogLocator(base=10.0, subs=(1.0,), numticks=20)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=200)

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(False)

    ax.grid(which="major", axis="x", color="0.75", linewidth=1.2, alpha=0.9)
    ax.grid(which="minor", axis="x", color="0.85", linewidth=0.9, alpha=0.8)

    ax.yaxis.grid(False)


def _legend_bottom(ncol=2):
    ax = plt.gca()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=ncol,
        borderaxespad=0.0,
    )


def _finalize_and_save(fname: str):
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# 5) Plot functions

def plot_article_x1_x5(res, fname):
    tmin = _log_safe_time(res["t_min"])
    Yexp = res["Yexp"]
    Yfit = res["Yfit"]

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)

    line_x1, = ax.plot(tmin, Yfit[:, 0], linewidth=3.2, label="x1 (model)")
    c1 = line_x1.get_color()
    ax.plot(
        tmin, Yexp[:, 0],
        linestyle="None", marker="o", markersize=8,
        markerfacecolor="none", markeredgecolor=c1, markeredgewidth=2.0,
        label="x1 (experimental)"
    )

    line_x5, = ax.plot(tmin, Yfit[:, 4], linewidth=3.2, label="x5 (model)")
    c5 = line_x5.get_color()
    ax.plot(
        tmin, Yexp[:, 4],
        linestyle="None", marker="o", markersize=8,
        markerfacecolor="none", markeredgecolor=c5, markeredgewidth=2.0,
        label="x5 (experimental)"
    )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Volumetric fraction (%)")

    apply_mastersizer_grid(ax)

    _legend_bottom(ncol=2)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_article_x2_x3_x4(res, fname):
    tmin = _log_safe_time(res["t_min"])
    Yexp = res["Yexp"]
    Yfit = res["Yfit"]

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)

    for idx, lab in zip([1, 2, 3], ["x2", "x3", "x4"]):
        line, = ax.plot(tmin, Yfit[:, idx], linewidth=3.2, label=f"{lab} (model)")
        c = line.get_color()
        ax.plot(
            tmin, Yexp[:, idx],
            linestyle="None", marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=c, markeredgewidth=2.0,
            label=f"{lab} (experimental)"
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Volumetric fraction (%)")

    apply_mastersizer_grid(ax)

    _legend_bottom(ncol=3)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_article_all(res, fname):
    tmin = _log_safe_time(res["t_min"])
    Yexp = res["Yexp"]
    Yfit = res["Yfit"]

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)

    for idx, lab in zip([0, 1, 2, 3, 4], ["x1", "x2", "x3", "x4", "x5"]):
        line, = ax.plot(tmin, Yfit[:, idx], linewidth=3.2, label=f"{lab} (model)")
        c = line.get_color()
        ax.plot(
            tmin, Yexp[:, idx],
            linestyle="None", marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=c, markeredgewidth=2.0,
            label=f"{lab} (experimental)"
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Volumetric fraction (%)")

    apply_mastersizer_grid(ax)

    _legend_bottom(ncol=5)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_article_comparison_x5(res_u, res_m, fname):
    tu = _log_safe_time(res_u["t_min"])
    tm = _log_safe_time(res_m["t_min"])
    Yu_exp, Yu_fit = res_u["Yexp"], res_u["Yfit"]
    Ym_exp, Ym_fit = res_m["Yexp"], res_m["Yfit"]

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)

    line_u, = ax.plot(tu, Yu_fit[:, 4], linewidth=3.2, label="Ultrasound x5 (model)")
    cu = line_u.get_color()
    ax.plot(
        tu, Yu_exp[:, 4],
        linestyle="None", marker="o", markersize=8,
        markerfacecolor="none", markeredgecolor=cu, markeredgewidth=2.0,
        label="Ultrasound x5 (experimental)"
    )

    line_m, = ax.plot(tm, Ym_fit[:, 4], linewidth=3.2, label="Ball milling x5 (model)")
    cm = line_m.get_color()
    ax.plot(
        tm, Ym_exp[:, 4],
        linestyle="None", marker="o", markersize=8,
        markerfacecolor="none", markeredgecolor=cm, markeredgewidth=2.0,
        label="Ball milling x5 (experimental)"
    )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Volumetric fraction (%)")

    apply_mastersizer_grid(ax)

    _legend_bottom(ncol=2)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)


# 6) Main

def main():
    set_mpl_article_style()

    res_u = fit_model("ultrasound", ultrasound)
    res_m = fit_model("ball_mill", ball_mill)

    plot_article_x1_x5(res_u, "FIG_ARTICLE_ultrasound_x1x5.png")
    plot_article_x2_x3_x4(res_u, "FIG_ARTICLE_ultrasound_x2x3x4.png")
    plot_article_all(res_u, "FIG_ARTICLE_ultrasound_all.png")

    plot_article_x1_x5(res_m, "FIG_ARTICLE_ball_mill_x1x5.png")
    plot_article_x2_x3_x4(res_m, "FIG_ARTICLE_ball_mill_x2x3x4.png")
    plot_article_all(res_m, "FIG_ARTICLE_ball_mill_all.png")

    plot_article_comparison_x5(res_u, res_m, "FIG_ARTICLE_comparison_x5.png")

    print("Ultrasound: k_out_x1 (h^-1) =", res_u["k_out_x1"], "t1/2 (min) =", res_u["t_half_x1_min"])
    print("Ball milling: k_out_x1 (h^-1) =", res_m["k_out_x1"], "t1/2 (min) =", res_m["t_half_x1_min"])

    print("\nOK! Files generated.")


if __name__ == "__main__":

    main()
