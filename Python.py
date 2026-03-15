from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
from scipy.stats import gaussian_kde, norm


EXCEL_FILE = Path("Excel.xlsx")
SHEET_NAME = "country estimates"

HEADERS = [
    "iso3",
    "year",
    "sex",
    "age_est",
    "age_lower",
    "age_upper",
    "crude_est",
    "crude_lower",
    "crude_upper",
    "pop18plus",
]

ISO3_NAME_OVERRIDES = {
    "ARE": "United Arab Emirates",
    "KOR": "Republic of Korea",
    "SAU": "Saudi Arabia",
    "IRN": "Iran",
    "PAK": "Pakistan",
    "AFG": "Afghanistan",
    "GUY": "Guyana",
    "QAT": "Qatar",
    "IRQ": "Iraq",
    "KWT": "Kuwait",
    "LBN": "Lebanon",
    "CUB": "Cuba",
    "PAN": "Panama",
    "PRT": "Portugal",
}

TOP_GAP_LABEL_OFFSETS = {
    "Cuba": (0.8, 0.8),
    "Pakistan": (0.6, 0.7),
    "Iran": (0.5, 0.6),
    "Afghanistan": (0.7, 0.8),
    "Guyana": (0.6, 0.7),
}


def ensure_excel_file(path: Path) -> Path:
    if path.exists():
        print(f"Найден Excel-файл: {path.resolve()}")
        return path

    raise FileNotFoundError(
        f"Excel-файл не найден: {path.resolve()}\n"
        "Положите Excel.xlsx в ту же папку, что и .py"
    )


def iso3_to_name(code: str) -> str:
    if code in ISO3_NAME_OVERRIDES:
        return ISO3_NAME_OVERRIDES[code]
    try:
        country = pycountry.countries.get(alpha_3=code)
        return country.name if country else code
    except Exception:
        return code


def load_country_estimates(path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        skiprows=2,
        names=HEADERS,
        header=None,
    )
    df = df[df["year"] == 2022].copy()
    df["pct"] = df["age_est"] * 100
    df["country"] = df["iso3"].astype(str).map(iso3_to_name)
    return df


def both_sexes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["sex"] == "both sexes"].copy()


def males(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["sex"] == "male"].copy()


def females(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["sex"] == "female"].copy()


def apply_report_style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(
        True,
        axis=grid_axis,
        linestyle=(0, (1, 2)),
        linewidth=0.8,
        color="#D0D0D0",
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#555555")


def plot_hist_kde_normal(data: pd.DataFrame) -> None:
    x = data["pct"].to_numpy()
    bins = np.linspace(x.min(), x.max(), 10)
    xs = np.linspace(-0.3, 69.3, 600)
    kde = gaussian_kde(x)
    mu = x.mean()
    sigma = x.std(ddof=1)

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=130)
    fig.patch.set_facecolor("white")
    apply_report_style(ax, grid_axis="y")

    ax.hist(
        x,
        bins=bins,
        density=True,
        color="#b7cfde",
        edgecolor="#555555",
        linewidth=1.1,
        alpha=0.95,
        label="Гистограмма плотности",
        zorder=2,
    )
    ax.plot(
        xs,
        kde(xs),
        color="#1458A6",
        linewidth=2.2,
        label="Ядерная оценка плотности",
        zorder=3,
    )
    ax.plot(
        xs,
        norm.pdf(xs, loc=mu, scale=sigma),
        color="#D7191C",
        linestyle="--",
        linewidth=2.0,
        label="Аппроксимация N(μ,σ)",
        zorder=3,
    )

    ax.set_title(
        "Распределение распространённости недостаточной физической активности\n"
        "по странам мира, 2022 г. (оба пола)",
        fontsize=16,
        pad=8,
    )
    ax.set_xlabel("Распространённость, %", fontsize=12)
    ax.set_ylabel("Плотность", fontsize=12)
    ax.set_xlim(-4, 72.5)
    ax.set_ylim(0, 0.034)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14)
    plt.show()


def plot_boxplot_by_sex(df: pd.DataFrame) -> None:
    both = both_sexes(df)["pct"].to_numpy()
    male = males(df)["pct"].to_numpy()
    female = females(df)["pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=130)
    fig.patch.set_facecolor("white")
    apply_report_style(ax, grid_axis="y")

    ax.boxplot(
        [both, male, female],
        labels=["Оба пола", "Мужчины", "Женщины"],
        patch_artist=True,
        boxprops=dict(facecolor="#b6d7a8", edgecolor="#1f8f44", linewidth=1.2),
        whiskerprops=dict(color="#1f8f44", linewidth=1.2),
        capprops=dict(color="#1f8f44", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.8),
        flierprops=dict(
            marker="o",
            markerfacecolor="#e06666",
            markeredgecolor="#d62f2f",
            markersize=5,
            alpha=0.95,
        ),
    )

    ax.set_title("Сравнение распределений по полу, 2022 г.", fontsize=15, pad=8)
    ax.set_ylabel("Распространённость, %", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_ylim(-1, 77)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
    plt.show()


def plot_top10_barh(data: pd.DataFrame) -> None:
    top10 = data.sort_values("pct", ascending=False).head(10).copy()
    top10 = top10.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=130)
    fig.patch.set_facecolor("white")
    apply_report_style(ax, grid_axis="x")

    bars = ax.barh(
        top10["country"],
        top10["pct"],
        color="#efaa66",
        edgecolor="black",
        linewidth=1.1,
        zorder=2,
    )

    ax.set_title(
        "Страны с наибольшей распространённостью недостаточной физической активности, 2022 г.",
        fontsize=14,
        pad=8,
    )
    ax.set_xlabel("Распространённость, %", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, 74)

    for bar, value in zip(bars, top10["pct"]):
        ax.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=9,
        )

    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.14)
    plt.show()


def plot_scatter_male_female(df: pd.DataFrame) -> None:
    male = males(df)[["iso3", "country", "pct"]].rename(columns={"pct": "male_pct"})
    female = females(df)[["iso3", "country", "pct"]].rename(columns={"pct": "female_pct"})
    merged = male.merge(female, on=["iso3", "country"], how="inner")
    merged["gap"] = merged["female_pct"] - merged["male_pct"]

    fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=130)
    fig.patch.set_facecolor("white")
    apply_report_style(ax, grid_axis="both")

    ax.scatter(
        merged["male_pct"],
        merged["female_pct"],
        s=42,
        color="#7e6ab1",
        alpha=0.88,
        edgecolors="#ffffff",
        linewidths=0.35,
        zorder=3,
    )

    line_min = min(merged["male_pct"].min(), merged["female_pct"].min())
    line_max = max(merged["male_pct"].max(), merged["female_pct"].max())
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        color="black",
        linewidth=1.2,
        zorder=2,
    )

    top_gap = merged.sort_values("gap", ascending=False).head(5)
    for _, row in top_gap.iterrows():
        dx, dy = TOP_GAP_LABEL_OFFSETS.get(row["country"], (0.6, 0.6))
        ax.text(row["male_pct"] + dx, row["female_pct"] + dy, row["country"], fontsize=9)

    ax.set_title(
        "Сопоставление страновых значений: мужчины и женщины, 2022 г.",
        fontsize=14,
        pad=8,
    )
    ax.set_xlabel("Мужчины, %", fontsize=12)
    ax.set_ylabel("Женщины, %", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(-2, 78)
    ax.set_ylim(-2, 78)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.12)
    plt.show()


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    excel_path = ensure_excel_file(EXCEL_FILE)
    df = load_country_estimates(excel_path)
    both = both_sexes(df)

    plot_hist_kde_normal(both)
    plot_boxplot_by_sex(df)
    plot_top10_barh(both)
    plot_scatter_male_female(df)


if __name__ == "__main__":
    main()