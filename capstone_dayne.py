import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, root_mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Union

from sklearn.svm import SVC

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


def preprocess(full_data, num_cols, qual_cols, tag_cols, *, thres=4, normalize_tag="tag_sum") -> pd.DataFrame:
    df = full_data[full_data["is_male"] != full_data["is_female"]]
    df = df[df["average_ratings"].notnull()]
    df_thres = df[df["num_ratings"] >= thres]

    if normalize_tag is None:
        return df_thres

    df_thres[tag_cols] = df_thres[tag_cols].astype("float64")

    if normalize_tag == "tag_sum":
        denom = df_thres[tag_cols].sum(axis=1)
    elif normalize_tag == "num_ratings":
        denom = full_data["num_ratings"]

    df_thres.loc[:, tag_cols] = df_thres[tag_cols].div(denom, axis=0)

    return df_thres


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


def build_regression_model(X, y, test_size=0.2, with_feature_scaling=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
    model = LinearRegression()

    if with_feature_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (model, rmse, r2, y_pred, y_test, scaler) if with_feature_scaling else (model, rmse, r2, y_pred, y_test)


def plot_regression_scatter(y_hat, y_true, target_name):
    fig, ax = plt.subplots()

    ax.scatter(x=y_hat, y=y_true, c="purple")
    ax.set_title(f"Scatterplot of {target_name} vs Predicted {target_name} (y_hat)")
    ax.set_xlabel(f"Predicted {target_name} (y_hat)")
    ax.set_ylabel(f"Actual {target_name} (y)")

    plt.show()


def sort_regress_model_coefs(coefs, feature_names):
    sorted_coefs = sorted(list(zip(coefs, feature_names)), key=lambda e: e[0], reverse=True)

    return sorted_coefs


def build_classification_model(X, y, test_size=0.2, threshold=0.5, model_type="logistic"):
    # default thresholds sbould be 0.5 for "logistic regression" and 0.0 for "svm"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logistic":
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:,1]
    elif model_type == "svm":
        model = SVC(kernel='linear', random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)

        y_prob = model.decision_function(X_test_scaled)

    model.fit(X_train_scaled, y_train)

    # y_prob = model.predict_proba(X_test_scaled)[:,1]
    # y_pred = model.predict(X_test_scaled)

    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, thres = roc_curve(y_test, y_prob)

    auc_score = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)


    return model, (auc_score, acc, precision, recall, f1), conf_matrix, (fpr, tpr, thres), (y_test, y_pred)


def plot_roc_curve(fpr, tpr, auc_score):
    # Calculate the ROC curve and AUC

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.show()


def q1(male_rating, female_rating):
    t_stat, p_val = stats.ttest_ind(male_rating, female_rating, equal_var=False)
    print("Q1. p-value: ", p_val, "t-statistic: ", t_stat)

    u_stat, p_val = stats.mannwhitneyu(male_rating, female_rating)
    print("Q1. p-value: ", p_val, "U statistic: ", u_stat)


def q2(male_rating, female_rating):
    rslt = stats.permutation_test((male_rating, female_rating), var_diff, n_resamples=1e+4, vectorized=True)
    print("Q2. p-value: ", rslt.pvalue, "variance difference: ", rslt.statistic)

    plt.hist(rslt.null_distribution, bins=50, density=True, label='Null Distribution')
    plt.axvline(rslt.statistic, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.title("Permutation Test Null Distribution")
    plt.xlabel("Test Statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show()



def q3(male_rating, female_rating):
    effect_size = mean_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=mean_diff_effect_size)
    plot_sampling_distribution(bs_results, effect_size, conf_interval, title="bootstrap result of mean difference average ratings")
    print(f"Q3. Effect size of gender bias in average ratings: {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")

    effect_size = var_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=var_diff_effect_size)

    plot_sampling_distribution(bs_results, effect_size, conf_interval, title="bootstrap result of variance difference in average ratings")
    print(f"Q3. Effect size of gender bias in spread of average ratings (variance difference): {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")



def q4(male_tag_df, female_tag_df):
    statistics = []
    p_vals = []

    tag_cols = male_tag_df.columns

    def perm_stat_fn(sample1, sample2, axis):
        return np.median(sample1, axis=axis) - np.median(sample2, axis=axis)

    for tag in tag_cols:
        male_tag = male_tag_df[tag]
        female_tag = female_tag_df[tag]

        rslt = stats.permutation_test((male_tag, female_tag), statistic=perm_stat_fn, n_resamples=1000, vectorized=True)

        statistics.append(rslt.statistic)
        p_vals.append(rslt.pvalue)

    p_vals_labeled = sorted(zip(p_vals, tag_cols), key=lambda e: e[0])

    top_k = 3
    alpha_level = 0.005
    sig_cnt = len([val for val in p_vals if val < alpha_level])

    print(f"Q4. The number of significant tags: {sig_cnt}")
    print(f"Q4. most gendered tags and corresponding p-values: {p_vals_labeled[:top_k+1]}")
    print(f"Q4. least gendered tags and corresponding p-values: {p_vals_labeled[::-1][:top_k+1]}")


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
    model, rmse, r2, _, _, _ = build_regression_model(features.to_numpy(), target.to_numpy())
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q7. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)


def q8(features: pd.DataFrame, target: pd.DataFrame):
    model, rmse, r2, _, _ = build_regression_model(features.to_numpy(), target.to_numpy(), with_feature_scaling=False)
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q8. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)



def q9(features: pd.DataFrame, target: pd.DataFrame):
    model, rmse, r2, _, _ = build_regression_model(features.to_numpy(), target.to_numpy(), with_feature_scaling=False)
    sorted_coefs = sort_regress_model_coefs(model.coef_, features.columns)

    print(f"Q9. model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("coefficients: ")
    for coef in sorted_coefs:
        print(coef)


def q10(X, y):
    classifier, (auc_score, acc, precision, recall, f1), conf_matrix, (fpr, tpr, thres), (y_test, y_pred) = build_classification_model(X, y, test_size=0.2, threshold=0.3, model_type="logistic")

    plot_roc_curve(fpr, tpr, auc_score)

    print("AUC Score:", auc_score)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def extra():
    pass


def main():
    np.random.seed(RANDOM_SEED)
    num_df, qual_df, tag_df, full_data = load_data("rmpCapstoneNum.csv", "rmpCapstoneQual.csv", "rmpCapstoneTags.csv")
    num_cols, qual_cols, tag_cols = num_df.columns, qual_df.columns, tag_df.columns
    processed_df = preprocess(full_data, num_cols, qual_cols, tag_cols, thres=4, normalize_tag="num_ratings")

    male_df = processed_df[processed_df["is_male"] == 1]
    female_df = processed_df[processed_df["is_female"] == 1]

    male_rating = male_df["average_ratings"]
    female_rating = female_df["average_ratings"]

    # q1(male_rating, female_rating)
    # q2(male_rating, female_rating)
    # q3(male_rating, female_rating)

    # q4(male_df[tag_cols], female_df[tag_cols])

    male_difficulty = male_df["average_difficulty"]
    female_difficulty = female_df["average_difficulty"]

    # q5(male_difficulty, female_difficulty)
    # q6(male_difficulty, female_difficulty)

    take_again_na_dropped_df = processed_df.dropna()

    na_dropped_num_df = take_again_na_dropped_df[num_cols]
    na_dropped_num_df = na_dropped_num_df.drop(columns="is_female")   # handle dummy variable trap

    numerical_features = na_dropped_num_df.drop(columns=["average_ratings"])
    target = take_again_na_dropped_df["average_ratings"]
    # q7(numerical_features, target)

    tag_features = take_again_na_dropped_df[tag_cols]

    # q8(tag_features, target)

    tag_features = processed_df[tag_cols]
    target = processed_df["average_difficulty"]
    # q9(tag_features, target)

    num_tag_df = take_again_na_dropped_df[list(num_cols) + list(tag_cols)]

    features = num_tag_df.drop(columns=["is_received_pepper", "is_female"])
    target = num_tag_df["is_received_pepper"]
    q10(features, target)

    # extra()


if __name__ == "__main__":
    main()
