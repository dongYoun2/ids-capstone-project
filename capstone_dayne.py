import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 18038726


def load_data(num_csv="rmpCapstoneNum.csv", qual_csv="rmpCapstoneQual.csv", tag_csv="rmpCapstoneTags.csv"):
    num_columns = [
        "average_ratings",
        "average_difficulty",
        "num_ratings",
        "is_received_pepper",
        "prop_take_again",
        "num_ratings_online",
        "is_male",
        "is_female",
    ]
    qual_columns = ["Major", "University", "State"]
    tag_columns = [
        "Tough grader",
        "Good feedback",
        "Respected",
        "Lots to read",
        "Participation matters",
        "Don't skip class or you will not pass",
        "Lots of homework",
        "Inspirational",
        "Pop quizzes!",
        "Accessible",
        "So many papers",
        "Clear grading",
        "Hilarious",
        "Test heavy",
        "Graded by few things",
        "Amazing lectures",
        "Caring",
        "Extra credit",
        "Group projects",
        "Lecture heavy",
    ]

    rmp_num_df = pd.read_csv(num_csv, header=None, names=num_columns)
    rmp_qual_df = pd.read_csv(qual_csv, header=None, names=qual_columns)
    rmp_tag_df = pd.read_csv(tag_csv, header=None, names=tag_columns)

    full_dat = rmp_num_df.join(rmp_tag_df).join(rmp_qual_df)

    return rmp_num_df, rmp_qual_df, rmp_tag_df, full_dat


def preprocess(full_data, *, thres=4):
    gender_grouped_df = full_data[full_data["is_male"] != full_data["is_female"]]

    male_df = gender_grouped_df[gender_grouped_df["is_male"] == 1]
    female_df = gender_grouped_df[gender_grouped_df["is_female"] == 1]

    rating_null_dropped_df = gender_grouped_df[gender_grouped_df['average_ratings'].notnull()]
    df_thres = rating_null_dropped_df[rating_null_dropped_df["num_ratings"] >= thres]

    male_df_thres = df_thres[df_thres["is_male"] == 1]
    female_df_thres = df_thres[df_thres["is_female"] == 1]

    return gender_grouped_df, (male_df, female_df), df_thres, (male_df_thres, female_df_thres)


def var_diff(sample1, sample2, axis=0):
    var1 = np.var(sample1, ddof=1, axis=axis)
    var2 = np.var(sample2, ddof=1, axis=axis)

    return var1 - var2


def pooled_std(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(sample1) + (n2 - 1) * np.var(sample2)) / (n1 + n2 - 2))

    return pooled_std


def cohens_d(sample1, sample2):
    mean1, mean2 = np.mean(sample1), np.mean(sample2)

    return (mean1 - mean2) / pooled_std(sample1, sample2)


def mean_diff_effect_size(sample1, sample2):
    effect_size = abs(cohens_d(sample1, sample2))

    return effect_size


def var_diff_effect_size(sample1, sample2):
    effect_size = abs(var_diff(sample1, sample2)) / pooled_std(sample1, sample2)

    return effect_size


def bootstrap(sample1, sample2, stat_fn, n_exp=10000):
    bs_estimates = []

    for _ in range(n_exp):
        bs_sample1 = np.random.choice(sample1, len(sample1))
        bs_sample2 = np.random.choice(sample2, len(sample2))

        bs_estimates.append(stat_fn(bs_sample1, bs_sample2))

    lower_bound = np.percentile(bs_estimates, q=2.5)
    upper_bound = np.percentile(bs_estimates, q=97.5)

    mean_estimate = np.mean(bs_estimates)

    conf_interval = (lower_bound, upper_bound)

    return conf_interval, mean_estimate, bs_estimates


def plot_sampling_distribution(bootstrap_results, observed_effect, conf_interval, title):
    lower_bound, upper_bound = conf_interval

    plt.hist(bootstrap_results, bins=50)
    plt.axvline(lower_bound, color='r', linestyle='dashed',
                linewidth=1.5, label='2.5% Bound')
    plt.axvline(upper_bound, color='r', linestyle='dashed',
                linewidth=1.5, label='97.5% Bound')
    plt.axvline(observed_effect, color='y', linestyle='solid',
                linewidth=1.5, label='observed effect size')

    # Adding labels and title
    plt.xlabel('Effect Size')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()


def build_regression_model(X, y, val_size=0.2, test_size=0.2):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    rmse_val_list = []
    models = []
    alphas = np.arange(0.01, 1, 0.01)
    for alpha in alphas:
        model = Lasso(alpha)
        model.fit(X_train_scaled, y_train)

        y_pred_val = model.predict(X_val_scaled)
        rmse = root_mean_squared_error(y_val, y_pred_val)

        rmse_val_list.append(rmse)
        models.append(model)

    min_i = np.argmin(np.array(rmse_val_list))
    best_model = models[min_i]

    y_pred_test = best_model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    return best_model, rmse, r2, scaler


def sort_regress_model_coefs(coefs, feature_names):
    sorted_coefs = sorted(list(zip(coefs, feature_names)), key=lambda e: e[0], reverse=True)

    return sorted_coefs


def impute_prop_take_again(num_df: pd.DataFrame):
    feature_df = num_df.drop(columns=["is_female", "prop_take_again"])  # handle dummpy variable trap (is_male, is_femlae)
    target_df = num_df["prop_take_again"]

    X_train = feature_df[target_df.notnull()].to_numpy()
    y_train = target_df[target_df.notnull()].to_numpy()

    model, _, _, feature_scaler = build_regression_model(X_train, y_train, val_size=0.2, test_size=0.2)

    X_to_impute = feature_df[target_df.isna()].to_numpy()

    y_pred = model.predict(feature_scaler.transform(X_to_impute))
    imputed_num_df = num_df.copy(deep=True)
    imputed_num_df.loc[target_df.isnull(), "prop_take_again"] = y_pred

    return imputed_num_df


def q1(male_rating, female_rating):
    t_stat, p_val = stats.ttest_ind(male_rating, female_rating, equal_var=False)
    print("Q1. p-value: ", p_val, "t-statistic: ", t_stat)

    u_stat, p_val = stats.mannwhitneyu(male_rating, female_rating)
    print("Q1. p-value: ", p_val, "U statistic: ", u_stat)


def q2(male_rating, female_rating):
    rslt = stats.permutation_test((male_rating, female_rating), var_diff, n_resamples=1e+4, vectorized=True)
    print("Q2. p-value: ", rslt.pvalue, "variance ratio: ", rslt.statistic)


def q3(male_rating, female_rating):
    effect_size = mean_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=mean_diff_effect_size)
    print(f"Q3. Effect size of gender bias in average ratings: {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")

    effect_size = var_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=var_diff_effect_size)

    plot_sampling_distribution(bs_results, effect_size, conf_interval, title="bootstrap result of effect size (variance difference) average ratings")

    print(f"Q3. Effect size of gender bias in spread of average ratings (variance difference): {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")


def q4(male_tag_df, female_tag_df):
    statistics = []
    p_vals = []

    for tag in male_tag_df.columns:
        male_df = male_tag_df[tag]
        female_df = female_tag_df[tag]

        statistic, p_val = stats.mannwhitneyu(male_df, female_df)
        statistics.append(statistic)
        p_vals.append(p_val)

    top_k = 3
    p_vals_np = np.array(p_vals)
    sorted_col_indicies = np.argsort(p_vals_np)
    min_indices = sorted_col_indicies[:top_k]
    max_indicies = sorted_col_indicies[-top_k:len(p_vals)]

    biased_cols = [male_tag_df.columns[min_indices[i]] for i in range(top_k)]
    unbiased_cols = [male_tag_df.columns[max_indicies[i]] for i in range(top_k)]

    min_p_vals = p_vals_np[min_indices]
    max_p_vals = p_vals_np[max_indicies]

    print(f"Q4. most gendered tags and corresponding p-values: {biased_cols}, {min_p_vals}")
    print(f"Q4. least gendered tags and corresponding p-values: {unbiased_cols}, {max_p_vals}")


def q5(male_rating, female_rating):
    t_stat, p_val = stats.ttest_ind(male_rating, female_rating, equal_var=False)
    print("Q5. p-value: ", p_val, "t-statistic: ", t_stat)

    u_stat, p_val = stats.mannwhitneyu(male_rating, female_rating)
    print("Q5. p-value: ", p_val, "U statistic: ", u_stat)


def q6(male_difficulty, female_difficulty):
    effect_size = mean_diff_effect_size(male_difficulty, female_difficulty)
    conf_interval, _, bs_results = bootstrap(male_difficulty, female_difficulty, stat_fn=mean_diff_effect_size)
    print(f"Q6. Effect size of gender bias in average difficulty: {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")

    plot_sampling_distribution(bs_results, effect_size, conf_interval, title="bootstrap result of effect size (mean difference) in average difficulty")


def q7(features: pd.DataFrame, target: pd.DataFrame):
    model, rmse, r2, _ = build_regression_model(features.to_numpy(), target.to_numpy())
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q7. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)


def q8(features: pd.DataFrame, target: pd.DataFrame):
    model, rmse, r2, _ = build_regression_model(features.to_numpy(), target.to_numpy())
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q8. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)



def q9(features: pd.DataFrame, target: pd.DataFrame):
    model, rmse, r2, _ = build_regression_model(features.to_numpy(), target.to_numpy())
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q9. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)


def q10():
    pass


def extra():
    pass


def main():
    np.random.seed(RANDOM_SEED)
    num_df, qual_df, tag_df, full_data = load_data("rmpCapstoneNum.csv", "rmpCapstoneQual.csv", "rmpCapstoneTags.csv")
    processed_df, (male_df, female_df), df_thres, (male_df_thres, female_df_thres) = preprocess(full_data, thres=4)

    male_rating_df = male_df_thres["average_ratings"]
    female_rating_df = female_df_thres["average_ratings"]

    male_difficulty_df = male_df_thres["average_difficulty"]
    female_difficulty_df = female_df_thres["average_difficulty"]

    q1(male_rating_df, female_rating_df)
    q2(male_rating_df, female_rating_df)
    q3(male_rating_df, female_rating_df)
    q4(male_df[tag_df.columns], female_df[tag_df.columns])
    q5(male_difficulty_df, female_difficulty_df)
    q6(male_difficulty_df, female_difficulty_df)

    num_df_thres = df_thres[num_df.columns]
    imputed_num_df_thres = impute_prop_take_again(num_df_thres)

    imputed_num_df_thres.drop(columns=["is_female"], inplace=True)

    features = imputed_num_df_thres.drop(columns=["average_ratings"])
    target = imputed_num_df_thres["average_ratings"]
    q7(features, target)

    features = df_thres[tag_df.columns]
    q8(features, target)

    target = df_thres["average_difficulty"]
    q9(features, target)
    q10()
    extra()


if __name__ == "__main__":
    main()
