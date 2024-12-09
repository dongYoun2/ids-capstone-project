import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, root_mean_squared_error, r2_score, accuracy_score
from sklearn.base import BaseEstimator

from typing import Optional, Literal, Sequence, Callable

RANDOM_SEED = 18038726

def load_data(num_csv="rmpCapstoneNum.csv", qual_csv="rmpCapstoneQual.csv", tag_csv="rmpCapstoneTags.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the numerical, qualitative, and tag data from CSV files and combines them into a single DataFrame.

    Args:
        num_csv (str, optional): Path to the numerical data CSV file. Defaults to "rmpCapstoneNum.csv".
        qual_csv (str, optional): Path to the qualitative data CSV file. Defaults to "rmpCapstoneQual.csv".
        tag_csv (str, optional): Path to the tag data CSV file. Defaults to "rmpCapstoneTags.csv".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: (numerical data DataFrame, qualitative data DataFrame, tag data DataFrame, combined DataFrame)
    """
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

    # Combine the datasets to create the full dataset
    full_dat = rmp_num_df.join(rmp_tag_df).join(rmp_qual_df)

    return rmp_num_df, rmp_qual_df, rmp_tag_df, full_dat


def preprocess(full_data: pd.DataFrame, num_cols: Sequence[str], qual_cols: Sequence[str], tag_cols: Sequence[str], *, thres=4, normalize_tag: Optional[Literal["tag_sum", "num_ratings"]] = None) -> pd.DataFrame:
    """
    Preprocesses the combined data by filtering, cleaning, and normalizing tag columns.

    Args:
        full_data (pd.DataFrame): The combined dataset.
        num_cols (Sequence[str]): List of numerical column names.
        qual_cols (Sequence[str]): List of qualitative column names.
        tag_cols (Sequence[str]): List of tag column names.
        thres (int, optional): Minimum number of ratings for filtering. Defaults to 4.
        normalize_tag (Optional[Literal["tag_sum", "num_ratings"]], optional): Normalization strategy for tag columns. Defaults to None.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Filter rows where gender is ambiguous or ratings are missing
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

    # Normalize tag columns if requested
    df_thres.loc[:, tag_cols] = df_thres[tag_cols].div(denom, axis=0)

    return df_thres


def var_diff(sample1: np.ndarray, sample2: np.ndarray, axis=0) -> float:
    """
    Computes the variance difference between two samples.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.
        axis (int, optional): Numpy axis for variance calculation. Defaults to 0.

    Returns:
        float: Variance difference between the two samples.
    """
    var1 = np.var(sample1, ddof=1, axis=axis)
    var2 = np.var(sample2, ddof=1, axis=axis)

    return var1 - var2


def pooled_std(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Computes the pooled standard deviation for two samples.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.

    Returns:
        float: Pooled standard deviation.
    """
    n1, n2 = len(sample1), len(sample2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(sample1) + (n2 - 1) * np.var(sample2)) / (n1 + n2 - 2))

    return pooled_std


def cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Computes Cohen's d between two samples.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.

    Returns:
        float: Cohen's d value.
    """
    mean1, mean2 = np.mean(sample1), np.mean(sample2)

    return (mean1 - mean2) / pooled_std(sample1, sample2)


def mean_diff_effect_size(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Computes the effect size based on the mean difference (cohen's d) between two samples.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.

    Returns:
        float: Effect size of the mean difference.
    """
    effect_size = abs(cohens_d(sample1, sample2))

    return effect_size


def var_diff_effect_size(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Computes the effect size based on the variance difference (analogous to cohen's d) between two samples.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.

    Returns:
        float: Effect size of the variance difference.
    """
    effect_size = abs(var_diff(sample1, sample2)) / pooled_std(sample1, sample2)

    return effect_size


def bootstrap(sample1: np.ndarray, sample2: np.ndarray, stat_fn: Callable[[np.ndarray, np.ndarray], float], n_exp=10000) -> tuple[tuple[float, float], float, list[float]]:
    """
    Performs bootstrap resampling to compute confidence intervals and mean estimates.

    Args:
        sample1 (np.ndarray): First sample.
        sample2 (np.ndarray): Second sample.
        stat_fn (Callable[[np.ndarray, np.ndarray], float]): Test statistic to compute on resampled data.
        n_exp (int, optional): umber of bootstrap iterations. Defaults to 10000.

    Returns:
        tuple[tuple[float, float], float, list[float]]: Confidence interval, mean estimate, and bootstrap results.
    """
    bs_estimates = []

    for _ in range(n_exp):
        # Randomly sample with replacement
        bs_sample1 = np.random.choice(sample1, len(sample1))
        bs_sample2 = np.random.choice(sample2, len(sample2))

        bs_estimates.append(stat_fn(bs_sample1, bs_sample2))

    lower_bound = np.percentile(bs_estimates, q=2.5)
    upper_bound = np.percentile(bs_estimates, q=97.5)

    mean_estimate = np.mean(bs_estimates)

    conf_interval = (lower_bound, upper_bound)

    return conf_interval, mean_estimate, bs_estimates


def plot_sampling_distribution(bootstrap_results: list[float], observed_effect: float, conf_interval: tuple[float, float], title: str):
    """
    Plots the bootstrap sampling distribution with observed effect size and confidence intervals.

    Args:
        bootstrap_results (list[float]): List of bootstrap estimates.
        observed_effect (float): Observed effect size.
        conf_interval (tuple[float, float]): Confidence interval for the bootstrap estimates.
        title (str): Title of the plot.
    """
    lower_bound, upper_bound = conf_interval

    plt.hist(bootstrap_results, bins=50)
    plt.axvline(lower_bound, color='r', linestyle='dashed',
                linewidth=1.5, label='2.5% Bound')
    plt.axvline(upper_bound, color='r', linestyle='dashed',
                linewidth=1.5, label='97.5% Bound')
    # Add observed effect size line
    plt.axvline(observed_effect, color='y', linestyle='solid',
                linewidth=1.5, label='observed effect size')

    # Adding labels and title
    plt.xlabel('Effect Size')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()


def build_regression_model(X: np.ndarray, y: np.ndarray, test_size=0.2, with_feature_scaling=True, random_seed=RANDOM_SEED) -> tuple[LinearRegression, float, float, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Builds and evaluates a linear regression model.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target variable.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
        with_feature_scaling (bool, optional): Whether to scale features. Defaults to True.
        random_seed (int, optional): Seed for reproducibility. Defaults to RANDOM_SEED.

    Returns:
        tuple[LinearRegression, float, float, np.ndarray, np.ndarray, Optional[StandardScaler]]: Trained model, RMSE, R^2, predictions, test targets, and scaler (if applied).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    model = LinearRegression()

    scaler = None

    if with_feature_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2, y_pred, y_test, scaler


def draw_regression_scatter_plot(y_hat: np.ndarray, y_true: np.ndarray, target_name: str):
    """
    Creates a scatter plot comparing predicted values (y_hat) and actual values (y_true) for regression analysis.

    Args:
        y_hat (np.ndarray): Predicted values from the regression model.
        y_true (np.ndarray): Actual values (ground truth).
        target_name (str): Name of the target variable for labeling the axes.
    """
    fig, ax = plt.subplots()

    ax.scatter(x=y_hat, y=y_true, c="purple")
    ax.set_title(f"Scatterplot of {target_name} vs Predicted {target_name} (y_hat)")
    ax.set_xlabel(f"Predicted {target_name} (y_hat)")
    ax.set_ylabel(f"Actual {target_name} (y)")

    plt.show()


def sort_regression_model_coefs(coefs: np.ndarray, feature_names: Sequence[str]) -> list[tuple[str, float]]:
    """
    Sorts regression model coefficients in descending order.

    Args:
        coefs (np.ndarray): Coefficients of the regression model.
        feature_names (Sequence[str]): predictor names corresponding to the coefficients.

    Returns:
        list[tuple[str, float]]: Sorted list of feature names and their coefficients.
    """
    sorted_coefs = sorted(list(zip(feature_names, coefs)), key=lambda e: e[1], reverse=True)

    return sorted_coefs


def build_classification_model(X: np.ndarray, y: np.ndarray, model_type: Literal["logistic", "svm"], threshold: float, test_size=0.2, random_seed=RANDOM_SEED) -> tuple[BaseEstimator, tuple[float, float, float, float, float], np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Builds and evaluates a classification model using logistic regression or SVM.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target variable.
        model_type (Literal["logistic", "svm"]): Type of model
        threshold (float): Threshold for logistic regression, decision value for support vector machine.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
        random_seed (int, optional): Seed for reproducibility. Defaults to RANDOM_SEED.

    Returns:
        tuple[BaseEstimator, tuple[float, float, float, float, float], np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: Trained odel, evaluation metrics (AUC, accuracy, precision, recall, F1), confusion matrix, ROC curve data (FPR, TPR, thresholds), and predictions (y_test, y_pred).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logistic":
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:,1]
    elif model_type == "svm":
        model = SVC(kernel='linear', random_state=random_seed)
        model.fit(X_train_scaled, y_train)

        y_prob = model.decision_function(X_test_scaled)

    model.fit(X_train_scaled, y_train)

    # Predict class labels based on threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    fpr, tpr, thres_arr = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, (auc_score, acc, precision, recall, f1), conf_matrix, (fpr, tpr, thres_arr), (y_test, y_pred)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, title="ROC Curve"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        auc_score (float): Area Under the ROC Curve (AUC) score.
        title (str, optional): Title of the plot. Defaults to "ROC Curve".
    """
    plt.figure(figsize=(8, 6))
    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    # Add random guess line
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.show()


def q1(male_rating: np.ndarray, female_rating: np.ndarray):
    """
    Performs statistical test (Welch's t-test) to compare average ratings of male and female professors.

    Args:
        male_rating (np.ndarray): Average ratings of male professors.
        female_rating (np.ndarray): Average ratings of female professors.
    """
    # Welch's t-test (for unequal variances)
    t_stat, p_val = stats.ttest_ind(male_rating, female_rating, equal_var=False)
    print("Q1. p-value: ", p_val, "t-statistic: ", t_stat)


def q2(male_rating: np.ndarray, female_rating: np.ndarray):
    """
    Conducts a permutation test to compare variance differences between average ratings of male and female professors.

    Args:
        male_rating (np.ndarray): Average ratings of male professors.
        female_rating (np.ndarray): Average ratings of female professors.
    """
    rslt = stats.permutation_test((male_rating, female_rating), statistic=var_diff, n_resamples=1e+4, vectorized=True)
    print("Q2. p-value: ", rslt.pvalue, "variance difference: ", rslt.statistic)

    # Plot null distribution
    plt.hist(rslt.null_distribution, bins=50, density=True, label='Null Distribution')
    plt.axvline(rslt.statistic, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.title("Permutation Test Null Distribution")
    plt.xlabel("Test Statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def q3(male_rating: np.ndarray, female_rating: np.ndarray):
    """
    Computes effect sizes for mean and variance differences in average ratings and performs bootstrap resampling.

    Args:
        male_rating (np.ndarray): Average ratings of male professors.
        female_rating (np.ndarray): Average ratings of female professors.
    """
    # Compute mean difference effect size
    observed_effect_size = mean_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=mean_diff_effect_size)

    plot_sampling_distribution(bs_results, observed_effect_size, conf_interval, title="bootstrap result of mean difference average ratings")
    print(f"Q3. Effect size of gender bias in average ratings: {observed_effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")

    # Compute variance difference effect size
    observed_effect_size = var_diff_effect_size(male_rating, female_rating)
    conf_interval, _, bs_results = bootstrap(male_rating, female_rating, stat_fn=var_diff_effect_size)

    plot_sampling_distribution(bs_results, observed_effect_size, conf_interval, title="bootstrap result of variance difference in average ratings")
    print(f"Q3. Effect size of gender bias in spread of average ratings (variance difference): {observed_effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")


def q4(male_tag_df: pd.DataFrame, female_tag_df: pd.DataFrame, tag_columns: Sequence[str]):
    """
    Performs a permutation test to analyze gender bias in tag data and identifies significant tags.

    Args:
        male_tag_df (pd.DataFrame): DataFrame of tags for male professors.
        female_tag_df (pd.DataFrame): DataFrame of tags for female professors.
        tag_columns (Sequence[str]): List of tag column names.
    """
    statistics = []
    p_vals = []

    # Compute the median difference as the test statistic
    def perm_stat_fn(sample1, sample2, axis):
        return np.median(sample1, axis=axis) - np.median(sample2, axis=axis)


    for tag in tag_columns:
        male_tag = male_tag_df[tag]
        female_tag = female_tag_df[tag]

        # Permutation test for each tag
        rslt = stats.permutation_test((male_tag, female_tag), statistic=perm_stat_fn, n_resamples=10000, vectorized=True)

        statistics.append(rslt.statistic)
        p_vals.append(rslt.pvalue)

    # Sort tags by their p-values for significance analysis
    p_vals_labeled = sorted(zip(tag_columns, p_vals), key=lambda e: e[1])

    top_k = 3
    alpha_level = 0.005
    sig_cnt = len([val for val in p_vals if val < alpha_level])

    print(f"Q4. The number of significant tags: {sig_cnt}")
    print(f"Q4. Most gendered tags and corresponding p-values: {p_vals_labeled[:top_k]}")
    print(f"Q4. Least gendered tags and corresponding p-values: {p_vals_labeled[::-1][:top_k]}")


def q5(male_difficulty: np.ndarray, female_difficulty: np.ndarray):
    """
    Performs statistical test (Welch's t-test) to assess gender bias in average difficulty.

    Args:
        male_difficulty (np.ndarray): Average difficulty for male professors.
        female_difficulty (np.ndarray): Average difficulty for female professors.
    """
    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(male_difficulty, female_difficulty, equal_var=False)
    print("Q5. p-value: ", p_val, "t-statistic: ", t_stat)


def q6(male_difficulty: np.ndarray, female_difficulty: np.ndarray):
    """
    Computes the effect size of gender bias in average difficulty and performs bootstrap resampling.

    Args:
        male_difficulty (np.ndarray): Average difficulty for male professors.
        female_difficulty (np.ndarray): Average difficulty for female professors.
    """
    # Compute effect size
    effect_size = mean_diff_effect_size(male_difficulty, female_difficulty)
    conf_interval, _, bs_results = bootstrap(male_difficulty, female_difficulty, stat_fn=mean_diff_effect_size)
    print(f"Q6. Effect size of gender bias in average difficulty: {effect_size} with confidence interval {float(conf_interval[0]), float(conf_interval[1])}")

    # Plot bootstrap results
    plot_sampling_distribution(bs_results, effect_size, conf_interval, title="bootstrap result of effect size (mean difference) in average difficulty")


def q7(X: np.ndarray, y: np.ndarray, predictors: Sequence[str]):
    """
    Builds a regression model to predict average ratings based on numerical data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable (average ratings).
        predictors (Sequence[str]): Feature names.
    """
    model, rmse, r2, y_hat, y_true, _ = build_regression_model(X, y)
    sorted_coefs = sort_regression_model_coefs(model.coef_, predictors)
    draw_regression_scatter_plot(y_hat, y_true, target_name="Average Ratings")

    print(f"Q7. Model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("Largest coefficient: ", sorted_coefs[0])


def q8(X: np.ndarray, y: np.ndarray, predictors: Sequence[str]):
    """
    Builds a regression model to predict average ratings based on tag data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable (average ratings).
        predictors (Sequence[str]): Feature names.
    """
    model, rmse, r2, y_hat, y_true, _ = build_regression_model(X, y, with_feature_scaling=False)
    sorted_coefs = sort_regression_model_coefs(model.coef_, predictors)
    draw_regression_scatter_plot(y_hat, y_true, target_name="Average Ratings")

    print(f"Q8. Model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("Largest coefficient: ", sorted_coefs[0])


def q9(X: np.ndarray, y: np.ndarray, predictors: Sequence[str]):
    """
    Builds a regression model to predict average difficulty based on tag features.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable (average difficulty).
        predictors (Sequence[str]): Feature names.
    """
    model, rmse, r2, y_hat, y_true, _ = build_regression_model(X, y, with_feature_scaling=False)
    sorted_coefs = sort_regression_model_coefs(model.coef_, predictors)
    draw_regression_scatter_plot(y_hat, y_true, target_name="Average Difficulty")

    print(f"Q9. Model: {model}, RMSE: {rmse}, R^2: {r2}")
    print("Largest coefficient: ", sorted_coefs[0])


def q10(X: np.ndarray, y: np.ndarray):
    """
    Builds classification models (logistic regression and support vector machine) to predict "is_received_pepper" indicator.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable (binary indicator for "is_received_pepper").
    """
    model_types = ["logistic", "svm"]
    thresholds = [0.3, 0]   # Decision thresholds for logistic regression and SVM
    plot_titles = ["ROC Curve of Logistic Regression", "ROC Curve of SVM"]

    for m_type, thres, title in zip(model_types, thresholds, plot_titles):
        classifier, (auc_score, acc, precision, recall, f1), conf_matrix, (fpr, tpr, thres_arr), (y_test, y_pred) = build_classification_model(X, y, model_type=m_type, threshold=thres, test_size=0.2)

        plot_roc_curve(fpr, tpr, auc_score, title=title)

        print("AUC Score:", auc_score)
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))


def extra_credit(samples: Sequence[np.ndarray]):
    """
    Performs statistical test (ANOVA) to examine difference in average difficulty ratings across states.

    Args:
        samples (Sequence[np.ndarray]): List of average difficulty of each state
    """
    test_stat, p_val = stats.f_oneway(*samples)

    print(f"Extra Credit: p-value: ", p_val, "test statistic: ", test_stat)


def main():
    np.random.seed(RANDOM_SEED)

    # Load and preprocess data
    num_df, qual_df, tag_df, full_data = load_data("rmpCapstoneNum.csv", "rmpCapstoneQual.csv", "rmpCapstoneTags.csv")
    num_cols, qual_cols, tag_cols = num_df.columns, qual_df.columns, tag_df.columns
    processed_df = preprocess(full_data, num_cols, qual_cols, tag_cols, thres=4, normalize_tag="num_ratings")

    # Separate data by gender
    male_df = processed_df[processed_df["is_male"] == 1]
    female_df = processed_df[processed_df["is_female"] == 1]

    # Perform statistical tests on ratings
    male_rating = male_df["average_ratings"]
    female_rating = female_df["average_ratings"]

    q1(male_rating.to_numpy(), female_rating.to_numpy())
    q2(male_rating. to_numpy(), female_rating.to_numpy())
    q3(male_rating.to_numpy(), female_rating.to_numpy())

    # Analyze gender bias in tags
    q4(male_df[tag_cols], female_df[tag_cols], tag_cols)

    # Perform statistical tests on difficulty ratings
    male_difficulty = male_df["average_difficulty"]
    female_difficulty = female_df["average_difficulty"]

    q5(male_difficulty.to_numpy(), female_difficulty.to_numpy())
    q6(male_difficulty.to_numpy(), female_difficulty.to_numpy())

    # Prepare data for regression analysis
    take_again_na_dropped_df = processed_df.dropna()
    take_again_na_dropped_num_df = take_again_na_dropped_df[num_cols].drop(columns="is_female")   # handle dummy variable trap

    # Regression model: numerical data -> average ratings
    numerical_features = take_again_na_dropped_num_df.drop(columns=["average_ratings"])
    target = take_again_na_dropped_df["average_ratings"]
    q7(numerical_features.to_numpy(), target.to_numpy(), numerical_features.columns)

    # Regression model: tag data -> average ratings
    tag_features = take_again_na_dropped_df[tag_cols]
    q8(tag_features.to_numpy(), target.to_numpy(), tag_features.columns)

    # Regression model: tag data -> average difficulty
    tag_features = processed_df[tag_cols]
    target = processed_df["average_difficulty"]
    q9(tag_features.to_numpy(), target.to_numpy(), tag_features.columns)

    # Classification models: numerical and tag data -> "is_received_pepper" indicator
    num_tag_df = take_again_na_dropped_df[list(num_cols) + list(tag_cols)]
    features = num_tag_df.drop(columns=["is_received_pepper", "is_female"])
    target = num_tag_df["is_received_pepper"]
    q10(features.to_numpy(), target.to_numpy())

    # Extra credit: ANOVA for state-wise average difficulty ratings
    state_num = 5
    state_df = processed_df["State"]
    state_cnt = state_df.value_counts(ascending=False).iloc[:state_num]
    states = state_cnt.index

    samples = []
    for state in states:
        state_df = processed_df[processed_df["State"] == state]
        state_rating_df = state_df["average_difficulty"]
        samples.append(state_rating_df.to_numpy())

    extra_credit(samples)


if __name__ == "__main__":
    main()
