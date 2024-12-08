# Library Import 
import numpy as np 
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import permutation_test,mannwhitneyu

# Random seed
np.random.seed(18038726)

# Reading in Files 
num = pd.read_csv('rmpCapstoneNum.csv', header=None)
tag = pd.read_csv('rmpCapstoneTags.csv', header=None)
qual = pd.read_csv('rmpCapstoneQual.csv', header=None)


# Renaming column names 
num_columns = ['average_ratings', 'average_difficulty', 'num_ratings', 'is_received_pepper', 'prop_take_again', 'num_ratings_online', 'is_male', 'is_female']
tag_columns = [ 'Tough grader', 'Good feedback', 'Respected', 'Lots to read', 'Participation matters', 'Don’t skip class or you will not pass', 'Lots of homework'
,'Inspirational','Pop quizzes!' ,'Accessible','So many papers' ,'Clear grading','Hilarious','Test heavy','Graded by few things','Amazing lectures','Caring','Extra credit'
,'Group projects' ,'Lecture heavy']
qual_columns = ['Major', 'University', 'State']

num.columns = num_columns
tag.columns = tag_columns 
qual.columns = qual_columns


# Preprocess 
full_dat = num.join(tag).join(qual)
num_na_drop = full_dat[full_dat['average_ratings'].notnull()]

# Threshold 
k = 4 
# Drop rows with professor's gender undetermined 
num_with_gender = num_na_drop[num_na_drop['is_male'] + num_na_drop['is_female'] == 1]
num_with_gender = num_with_gender[num_with_gender['num_ratings'] >= k]
num_with_gender['Gender'] = num_with_gender['is_male'].apply(lambda x: 'Male' if x == 1 else 'Female')
print(num_with_gender.is_male.sum(), num_with_gender.is_female.sum())

## ------------------------------------------------------------------------------------------------------------------------------
## Q1 
female_ratings = num_with_gender[num_with_gender['Gender'] == 'Female']['average_ratings']
male_ratings = num_with_gender[num_with_gender['Gender'] == 'Male']['average_ratings']

# T-test 
t_result = stats.ttest_ind(female_ratings, male_ratings, equal_var=False, alternative='two-sided')

# Mann-Whitney 
mann_whitney_result = mannwhitneyu(female_ratings, male_ratings, alternative='two-sided')

print(female_ratings.mean(), male_ratings.mean())
print(t_result)
print(mann_whitney_result)


## ------------------------------------------------------------------------------------------------------------------------------
## Q2 
observed_diff = np.var(male_ratings, ddof=1) - np.var(female_ratings, ddof=1)

def variance_diff(group1, group2):
    return np.var(group1, ddof=1) - np.var(group2, ddof=1)

# Permutation Test 
result = permutation_test(
    (male_ratings, female_ratings),
    statistic=variance_diff,
    vectorized=False,
    n_resamples=10000,
    alternative="two-sided",
)

# Results
print("Observed Difference:", observed_diff)
print("P-value:", result.pvalue)

null_distribution = result.null_distribution
observed_stat = result.statistic

# Plot the distribution
plt.hist(null_distribution, bins=30, density=True, alpha=0.7, color='blue', label='Null Distribution')
plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
plt.title("Permutation Test Null Distribution")
plt.xlabel("Test Statistic")
plt.ylabel("Density")
plt.legend()
plt.show()



## ------------------------------------------------------------------------------------------------------------------------------ 
## Q3

# block1 - calculating cohen's d for effect size of mean difference
def calculate_cohens_d(data1, data2):
    mean_difference = np.mean(data1) - np.mean(data2)
    pooled_standard_deviation = np.sqrt(
        (((len(data1) - 1) * np.var(data1, ddof=1)) +
         ((len(data2) - 1) * np.var(data2, ddof=1))) /
        (len(data1) + len(data2) - 2)
    )
    return abs(mean_difference) / pooled_standard_deviation

# block2 - calculating effect size of variance difference 
def compute_variance_effect(data1, data2):
    variance1 = np.var(data1, ddof=1)
    variance2 = np.var(data2, ddof=1)
    pooled_variance = (
        ((len(data1) - 1) * variance1 + (len(data2) - 1) * variance2) /
        (len(data1) + len(data2) - 2)
    )
    return abs(variance1 - variance2) / np.sqrt(pooled_variance)

# objective function - apply resampling method to constuct a sampling distribution of effect sizes 
def bootstrap_effect_size(data1, data2, metric="mean", num_resamples=10000, conf_level=0.95):
    if metric == "mean":
        effect_func = calculate_cohens_d
    elif metric == "variance":
        effect_func = compute_variance_effect
    
    observed_effect = effect_func(data1, data2)

    bootstrap_results = []
    for _ in range(num_resamples):
        resampled_data1 = np.random.choice(data1, size=len(data1), replace=True)
        resampled_data2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_results.append(effect_func(resampled_data1, resampled_data2))

    alpha = 1 - conf_level
    ci_lower = np.percentile(bootstrap_results, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_results, 100 * (1 - alpha / 2))

    return observed_effect, (ci_lower, ci_upper), bootstrap_results


mean_effect, mean_ci, mean_bootstrap = bootstrap_effect_size(male_ratings, female_ratings, metric="mean", num_resamples=10000)
print("Mean Difference (Cohen's d):")
print(f"Effect Size: {mean_effect:.3f}")
print(f"95% CI: {mean_ci}")


variance_effect, variance_ci, variance_bootstrap = bootstrap_effect_size(male_ratings, female_ratings, metric="variance", num_resamples=10000)
print("\nVariance Difference Effect Size:")
print(f"Effect Size: {variance_effect:.3f}")
print(f"95% CI: {variance_ci}")




# Plot confidence intervals 
def plot_sampling_distribution(bootstrap_results, observed_effect, confidence_interval, title="Effect Size Sampling Distribution"):
    ci_lower, ci_upper = confidence_interval

    plt.figure(figsize=(8, 5))
    plt.hist(bootstrap_results, bins=30, density=True, alpha=0.7, color='orange', label='Bootstrap Distribution')
    plt.axvline(observed_effect, color='red', linestyle='--', linewidth=2, label='Observed Effect')
    plt.axvline(ci_lower, color='green', linestyle='-', linewidth=1.5, label='95% CI Lower Bound')
    plt.axvline(ci_upper, color='green', linestyle='-', linewidth=1.5, label='95% CI Upper Bound')
    plt.title(title)
    plt.xlabel("Effect Size")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_sampling_distribution(mean_bootstrap, mean_effect, mean_ci, title="Mean Difference (Cohen's d)")
plot_sampling_distribution(variance_bootstrap, variance_effect, variance_ci, title="Variance Difference Effect Size")



## ------------------------------------------------------------------------------------------------------------------------------ 
# Q4 

num_with_gender_tag = num_with_gender[['Gender','Tough grader', 'Good feedback', 'Respected', 'Lots to read', 'Participation matters', 'Don’t skip class or you will not pass', 'Lots of homework'
,'Inspirational','Pop quizzes!' ,'Accessible','So many papers' ,'Clear grading','Hilarious','Test heavy','Graded by few things','Amazing lectures','Caring','Extra credit'
,'Group projects' ,'Lecture heavy', 'num_ratings']]

def normalize_tags(data, tag_columns):
    normalized_data = data.copy()
    for tag in tag_columns:
        normalized_data[tag] = normalized_data[tag] / normalized_data['num_ratings']
    return normalized_data
 
def mann_whitney_analysis(data, tag_column):
    male_data = data[data['Gender'] == 'Male'][tag_column]
    female_data = data[data['Gender'] == 'Female'][tag_column]
    
    stat, p_value = mannwhitneyu(male_data, female_data, alternative='two-sided')
    
    return {'Tag': tag_column, 'p-value': p_value}

num_with_gender_tag_norm = normalize_tags(num_with_gender_tag, tag_columns)

p_vals = {}
for t_column in tag_columns:
    result = mann_whitney_analysis(num_with_gender_tag_norm, t_column)
    p_vals[result['Tag']] = result['p-value']

print(len([v for k,v in p_vals.items() if v  < 0.005]))

pvals_sorted = sorted(p_vals.items(), key = lambda x: x[1])
print(pvals_sorted[:3])
print(pvals_sorted[-3:][::-1])


## ------------------------------------------------------------------------------------------------------------------------------ 
# Q5

female_diff = num_with_gender[num_with_gender['Gender'] == 'Female']['average_difficulty']
male_diff = num_with_gender[num_with_gender['Gender'] == 'Male']['average_difficulty']

# T-test 
t_result = stats.ttest_ind(female_diff, male_diff, equal_var=False, alternative='two-sided')

# Mann-Whitney 
mann_whitney_result = mannwhitneyu(female_diff, male_diff, alternative='two-sided')

print(female_diff.mean(), male_diff.mean())
print(t_result)
print(mann_whitney_result)



## ------------------------------------------------------------------------------------------------------------------------------ 
# Q6

# Q6 

mean_effect, mean_ci, mean_bootstrap = bootstrap_effect_size(male_diff, female_diff, metric="mean", num_resamples=10000)
print("Mean Difference (Cohen's d):")
print(f"Effect Size: {mean_effect:.3f}")
print(f"95% CI: {mean_ci}")


# Plot confidence intervals 
def plot_sampling_distribution(bootstrap_results, observed_effect, confidence_interval, title="Effect Size Sampling Distribution"):
    ci_lower, ci_upper = confidence_interval

    plt.figure(figsize=(8, 5))
    plt.hist(bootstrap_results, bins=30, density=True, alpha=0.7, color='orange', label='Bootstrap Distribution')
    plt.axvline(observed_effect, color='red', linestyle='--', linewidth=2, label='Observed Effect')
    plt.axvline(ci_lower, color='green', linestyle='-', linewidth=1.5, label='95% CI Lower Bound')
    plt.axvline(ci_upper, color='green', linestyle='-', linewidth=1.5, label='95% CI Upper Bound')
    plt.title(title)
    plt.xlabel("Effect Size")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_sampling_distribution(mean_bootstrap, mean_effect, mean_ci, title="Mean Difference (Cohen's d)")


## ------------------------------------------------------------------------------------------------------------------------------ 
# Q7


def linear_regression_analysis(data, target_column, predictor_columns, na_handling= None, na_field = None, test_size=0.2):
    # Handle NA values based on the selected option
    if na_handling == "Row Drop":
        data_selected = data.dropna()  # Drop rows with NA values
        y = data_selected[target_column]
        X = data_selected[predictor_columns]
    elif na_handling == "Column Drop":
        data_selected = data.copy()
        y = data_selected[target_column]
        X = data_selected[predictor_columns].drop([na_field], axis=1)  
    else:
        X = data[predictor_columns]
        y = data[target_column]
   
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    X_test = scaler.transform(X_test)
    y_pred_test = model.predict(X_test)
    
    r2 = r2_score(y_train, y_pred_train)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(len(X))
    for i in range(len(model.coef_)):
        print(f"{predictor_columns[i]} : {model.coef_[i]}")
    return r2, rmse, model.coef_

r2, rmse, coef = linear_regression_analysis(num_with_gender, target_column = 'average_ratings', predictor_columns = [x for x in num_columns if x not in  ['average_ratings', 'is_male']]
, na_handling= 'Row Drop', na_field = 'prop_take_again')

print(r2, rmse)


## ------------------------------------------------------------------------------------------------------------------------------ 
# Q8

dat = normalize_tags(num_with_gender[['is_female', 'num_ratings', 'average_ratings', 'prop_take_again'] + tag_columns], tag_columns).drop(['num_ratings'], axis = 1)

r2, rmse,coef = linear_regression_analysis(dat, target_column = 'average_ratings', predictor_columns=['is_female'] + tag_columns, na_handling= 'Row Drop', na_field = None, test_size=0.2)
print(r2, rmse)


## ------------------------------------------------------------------------------------------------------------------------------ 
# Q9 
dat = normalize_tags(num_with_gender[['is_female', 'num_ratings', 'average_difficulty'] + tag_columns], tag_columns).drop(['num_ratings'], axis = 1)

r2, rmse,coef = linear_regression_analysis(dat, target_column = 'average_difficulty', predictor_columns=['is_female'] + tag_columns, na_handling= 'Row Drop', na_field =  None, test_size=0.2)
print(r2, rmse)




## ------------------------------------------------------------------------------------------------------------------------------ 
# Q10 

def classifier(data, target_column, predictor_columns, test_size=0.2, mod = 'logistic', threshold = 0.3):
    X = data[predictor_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    if mod == 'logistic':
        model = LogisticRegression()
        model.fit(X_train, y_train)
    elif mod == 'svm':
        model = SVC(kernel='rbf', probability=True)  
        model.fit(X_train, y_train)
    elif mod == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=None)
        model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    X_test = scaler.transform(X_test)
    y_pred_test = model.predict(X_test)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # y_pred = model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)

  
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print('accuracy:', accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AU(RO)C: {auc:.3f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F-1 Score : {f1}")

  
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AU(RO)C = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

  
    return {
        "model": model,
        "AU(RO)C": auc,
        "confusion_matrix": conf_matrix,
        # "roc_curve": (fpr, tpr, thresholds),
        "f1 score" : f1,
        'accuracy' : accuracy,
        'precision': precision,
        'recall' : recall 
    }


dat = num_with_gender[num_columns + tag_columns].drop('is_male', axis = 1).dropna()
predictors = [x for x in dat.columns if x != 'is_received_pepper']
metrics= classifier(dat, target_column= 'is_received_pepper', predictor_columns= predictors, mod = 'logistic', threshold=0.5)

