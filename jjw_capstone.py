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