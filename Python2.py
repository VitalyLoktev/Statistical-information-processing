from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import chi2, kurtosis, norm, skew, t


excel_file = Path("Excel.xlsx")
sheet_name = "country estimates"

column_names = [
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

country_name_overrides = {
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
    "VNM": "Viet Nam",
}

published_global_estimate_pct = 31.30
significance_level = 0.05
number_of_intervals = 9


def get_country_name(iso3_code: str) -> str:
    if iso3_code in country_name_overrides:
        return country_name_overrides[iso3_code]
    country = pycountry.countries.get(alpha_3=iso3_code)
    return country.name if country is not None else iso3_code


def load_country_estimates(file_path: Path) -> pd.DataFrame:
    data = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=2,
        names=column_names,
        header=None,
    )
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["age_est"] = pd.to_numeric(data["age_est"], errors="coerce")
    data["age_lower"] = pd.to_numeric(data["age_lower"], errors="coerce")
    data["age_upper"] = pd.to_numeric(data["age_upper"], errors="coerce")
    data["pop18plus"] = pd.to_numeric(data["pop18plus"], errors="coerce")
    data["sex"] = data["sex"].astype(str).str.strip().str.lower()
    data = data[data["year"] == 2022].copy()
    data["country"] = data["iso3"].astype(str).map(get_country_name)
    data["pct"] = data["age_est"] * 100
    data["lower_pct"] = data["age_lower"] * 100
    data["upper_pct"] = data["age_upper"] * 100
    return data


def get_both_sexes(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["sex"] == "both sexes"].copy()


def get_males(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["sex"] == "male"].copy()


def get_females(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["sex"] == "female"].copy()


def calculate_main_statistics(values: pd.Series) -> dict:
    sample_size = int(values.size)
    mean_value = float(values.mean())
    median_value = float(values.median())
    first_quartile = float(values.quantile(0.25))
    third_quartile = float(values.quantile(0.75))
    minimum_value = float(values.min())
    maximum_value = float(values.max())
    value_range = maximum_value - minimum_value
    interquartile_range = third_quartile - first_quartile
    mean_absolute_deviation_from_median = float(np.mean(np.abs(values - median_value)))
    variance_value = float(values.var(ddof=1))
    standard_deviation_value = float(values.std(ddof=1))
    coefficient_of_variation = float(standard_deviation_value / mean_value * 100)
    skewness_value = float(skew(values.to_numpy(), bias=False))
    excess_kurtosis_value = float(kurtosis(values.to_numpy(), fisher=True, bias=False))

    return {
        "n": sample_size,
        "mean": mean_value,
        "median": median_value,
        "q1": first_quartile,
        "q3": third_quartile,
        "iqr": interquartile_range,
        "min": minimum_value,
        "max": maximum_value,
        "range": value_range,
        "mad_from_median": mean_absolute_deviation_from_median,
        "variance": variance_value,
        "standard_deviation": standard_deviation_value,
        "coefficient_of_variation": coefficient_of_variation,
        "skewness": skewness_value,
        "excess_kurtosis": excess_kurtosis_value,
    }


def build_grouped_series(values: pd.Series, interval_count: int):
    interval_edges = np.linspace(values.min(), values.max(), interval_count + 1)
    interval_width = float(interval_edges[1] - interval_edges[0])
    absolute_frequencies, _ = np.histogram(values.to_numpy(), bins=interval_edges)
    relative_frequencies = absolute_frequencies / absolute_frequencies.sum()
    reduced_frequencies = absolute_frequencies / (absolute_frequencies.sum() * interval_width)
    return interval_edges, absolute_frequencies, relative_frequencies, reduced_frequencies, interval_width


def calculate_chi_square_normality(
    values: pd.Series,
    mean_value: float,
    standard_deviation_value: float,
    interval_edges: np.ndarray,
    alpha: float,
) -> dict:
    grouped_observed = [
        int((values < interval_edges[1]).sum()),
        int(((values >= interval_edges[1]) & (values < interval_edges[2])).sum()),
        int(((values >= interval_edges[2]) & (values < interval_edges[3])).sum()),
        int(((values >= interval_edges[3]) & (values < interval_edges[4])).sum()),
        int(((values >= interval_edges[4]) & (values < interval_edges[5])).sum()),
        int(((values >= interval_edges[5]) & (values < interval_edges[6])).sum()),
        int(((values >= interval_edges[6]) & (values < interval_edges[7])).sum()),
        int((values >= interval_edges[7]).sum()),
    ]

    grouped_probabilities = [
        norm.cdf(interval_edges[1], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[2], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[1], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[3], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[2], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[4], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[3], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[5], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[4], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[6], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[5], loc=mean_value, scale=standard_deviation_value),
        norm.cdf(interval_edges[7], loc=mean_value, scale=standard_deviation_value)
        - norm.cdf(interval_edges[6], loc=mean_value, scale=standard_deviation_value),
        1 - norm.cdf(interval_edges[7], loc=mean_value, scale=standard_deviation_value),
    ]

    sample_size = int(values.size)
    expected_counts = np.array(grouped_probabilities) * sample_size
    chi_square_contributions = (np.array(grouped_observed) - expected_counts) ** 2 / expected_counts
    chi_square_statistic = float(chi_square_contributions.sum())
    degrees_of_freedom = len(grouped_observed) - 2 - 1
    critical_value = float(chi2.ppf(1 - alpha, degrees_of_freedom))
    p_value = float(chi2.sf(chi_square_statistic, degrees_of_freedom))

    return {
        "observed": grouped_observed,
        "expected": expected_counts,
        "contributions": chi_square_contributions,
        "chi_square_statistic": chi_square_statistic,
        "critical_value": critical_value,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
    }


def calculate_confidence_intervals(values: pd.Series, alpha: float) -> dict:
    sample_size = int(values.size)
    mean_value = float(values.mean())
    variance_value = float(values.var(ddof=1))
    standard_deviation_value = float(values.std(ddof=1))

    t_quantile = float(t.ppf(1 - alpha / 2, sample_size - 1))
    mean_margin = t_quantile * standard_deviation_value / np.sqrt(sample_size)
    mean_interval = (mean_value - mean_margin, mean_value + mean_margin)

    chi2_lower = float(chi2.ppf(alpha / 2, sample_size - 1))
    chi2_upper = float(chi2.ppf(1 - alpha / 2, sample_size - 1))

    variance_interval = (
        (sample_size - 1) * variance_value / chi2_upper,
        (sample_size - 1) * variance_value / chi2_lower,
    )
    standard_deviation_interval = (
        float(np.sqrt(variance_interval[0])),
        float(np.sqrt(variance_interval[1])),
    )

    return {
        "mean_interval": mean_interval,
        "standard_deviation_interval": standard_deviation_interval,
        "variance_interval": variance_interval,
    }


def calculate_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


def print_section(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    data = load_country_estimates(excel_file)

    both_sexes_data = get_both_sexes(data)
    male_data = get_males(data)
    female_data = get_females(data)

    both_sexes_values = both_sexes_data["pct"]
    male_values = male_data["pct"]
    female_values = female_data["pct"]

    main_statistics = calculate_main_statistics(both_sexes_values)
    male_statistics = calculate_main_statistics(male_values)
    female_statistics = calculate_main_statistics(female_values)

    interval_edges, absolute_frequencies, relative_frequencies, reduced_frequencies, interval_width = build_grouped_series(
        both_sexes_values,
        number_of_intervals,
    )

    chi_square_results = calculate_chi_square_normality(
        both_sexes_values,
        main_statistics["mean"],
        main_statistics["standard_deviation"],
        interval_edges,
        significance_level,
    )

    confidence_intervals = calculate_confidence_intervals(
        both_sexes_values,
        significance_level,
    )

    male_female_joined = male_data[["iso3", "country", "pct"]].rename(columns={"pct": "male_pct"}).merge(
        female_data[["iso3", "country", "pct"]].rename(columns={"pct": "female_pct"}),
        on=["iso3", "country"],
        how="inner",
    )
    male_female_joined["gap"] = male_female_joined["female_pct"] - male_female_joined["male_pct"]
    top_gender_gaps = male_female_joined.sort_values("gap", ascending=False).head(5)

    simple_mean_pct = float(both_sexes_values.mean())
    weighted_mean_pct = calculate_weighted_mean(both_sexes_data["pct"], both_sexes_data["pop18plus"])

    print_section("ОСНОВНЫЕ ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ")
    print(f"Объём выборки n = {main_statistics['n']}")
    print(f"Выборочное среднее = {main_statistics['mean']:.6f}")
    print(f"Медиана = {main_statistics['median']:.6f}")
    print(f"Первый квартиль Q1 = {main_statistics['q1']:.6f}")
    print(f"Третий квартиль Q3 = {main_statistics['q3']:.6f}")
    print(f"Межквартильный размах IQR = {main_statistics['iqr']:.6f}")
    print(f"Минимум = {main_statistics['min']:.6f}")
    print(f"Максимум = {main_statistics['max']:.6f}")
    print(f"Размах = {main_statistics['range']:.6f}")
    print(f"Среднее абсолютное отклонение от медианы = {main_statistics['mad_from_median']:.6f}")
    print(f"Выборочная дисперсия s^2 = {main_statistics['variance']:.6f}")
    print(f"Среднее квадратическое отклонение s = {main_statistics['standard_deviation']:.6f}")
    print(f"Коэффициент вариации = {main_statistics['coefficient_of_variation']:.6f}")
    print(f"Коэффициент асимметрии = {main_statistics['skewness']:.6f}")
    print(f"Эксцесс = {main_statistics['excess_kurtosis']:.6f}")

    print_section("ГРУППИРОВАННЫЙ СТАТИСТИЧЕСКИЙ РЯД")
    print(f"Число интервалов = {number_of_intervals}")
    print(f"Ширина интервала h = {interval_width:.6f}")
    for i in range(number_of_intervals):
        left_edge = interval_edges[i]
        right_edge = interval_edges[i + 1]
        absolute_frequency = int(absolute_frequencies[i])
        relative_frequency = float(relative_frequencies[i])
        reduced_frequency = float(reduced_frequencies[i])
        right_bracket = "]" if i == number_of_intervals - 1 else ")"
        print(
            f"Интервал [{left_edge:.2f}; {right_edge:.2f}{right_bracket}: "
            f"n_i = {absolute_frequency}, "
            f"w_i = {relative_frequency:.6f}, "
            f"n_i/(n*h) = {reduced_frequency:.6f}"
        )

    print_section("КРИТЕРИЙ ХИ-КВАДРАТ ПИРСОНА")
    for i in range(len(chi_square_results["observed"])):
        print(
            f"Группа {i + 1}: "
            f"наблюдаемая частота = {chi_square_results['observed'][i]}, "
            f"ожидаемая частота = {chi_square_results['expected'][i]:.6f}, "
            f"вклад в хи-квадрат = {chi_square_results['contributions'][i]:.6f}"
        )
    print(f"Наблюдаемое значение хи-квадрат = {chi_square_results['chi_square_statistic']:.6f}")
    print(f"Критическое значение хи-квадрат = {chi_square_results['critical_value']:.6f}")
    print(f"p-value = {chi_square_results['p_value']:.6f}")
    print(f"Число степеней свободы = {chi_square_results['degrees_of_freedom']}")

    print_section("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
    print(
        f"95%-й доверительный интервал для математического ожидания: "
        f"[{confidence_intervals['mean_interval'][0]:.6f}; {confidence_intervals['mean_interval'][1]:.6f}]"
    )
    print(
        f"95%-й доверительный интервал для стандартного отклонения: "
        f"[{confidence_intervals['standard_deviation_interval'][0]:.6f}; {confidence_intervals['standard_deviation_interval'][1]:.6f}]"
    )
    print(
        f"95%-й доверительный интервал для дисперсии: "
        f"[{confidence_intervals['variance_interval'][0]:.6f}; {confidence_intervals['variance_interval'][1]:.6f}]"
    )

    print_section("СРАВНЕНИЕ ПО ПОЛУ")
    print(
        f"Оба пола: n = {main_statistics['n']}, "
        f"среднее = {main_statistics['mean']:.6f}, "
        f"медиана = {main_statistics['median']:.6f}, "
        f"s = {main_statistics['standard_deviation']:.6f}, "
        f"Q1 = {main_statistics['q1']:.6f}, "
        f"Q3 = {main_statistics['q3']:.6f}, "
        f"максимум = {main_statistics['max']:.6f}"
    )
    print(
        f"Мужчины: n = {male_statistics['n']}, "
        f"среднее = {male_statistics['mean']:.6f}, "
        f"медиана = {male_statistics['median']:.6f}, "
        f"s = {male_statistics['standard_deviation']:.6f}, "
        f"Q1 = {male_statistics['q1']:.6f}, "
        f"Q3 = {male_statistics['q3']:.6f}, "
        f"максимум = {male_statistics['max']:.6f}"
    )
    print(
        f"Женщины: n = {female_statistics['n']}, "
        f"среднее = {female_statistics['mean']:.6f}, "
        f"медиана = {female_statistics['median']:.6f}, "
        f"s = {female_statistics['standard_deviation']:.6f}, "
        f"Q1 = {female_statistics['q1']:.6f}, "
        f"Q3 = {female_statistics['q3']:.6f}, "
        f"максимум = {female_statistics['max']:.6f}"
    )

    print_section("НАИБОЛЬШИЕ ГЕНДЕРНЫЕ РАЗЛИЧИЯ")
    for _, row in top_gender_gaps.iterrows():
        print(
            f"{row['country']}: "
            f"мужчины = {row['male_pct']:.6f}, "
            f"женщины = {row['female_pct']:.6f}, "
            f"разность = {row['gap']:.6f}"
        )

    print_section("ПРОСТОЕ, ВЗВЕШЕННОЕ И ГЛОБАЛЬНОЕ СРЕДНЕЕ")
    print(f"Простое среднее по странам = {simple_mean_pct:.6f}")
    print(f"Взвешенное среднее по взрослому населению = {weighted_mean_pct:.6f}")
    print(f"Глобальная оценка WHO = {published_global_estimate_pct:.6f}")


if __name__ == "__main__":
    main()