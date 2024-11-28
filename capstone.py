import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# %%
rmp_num_df = pd.read_csv("rmpCapstoneNum.csv")
rmp_qual_df = pd.read_csv("rmpCapstoneQual.csv")
rmp_tag_df = pd.read_csv("rmpCapstoneTags.csv")

rmp_num_df = rmp_num_df.rename(columns={
    "5": "average rating",
    "1.5": "average difficulty",
    "2": "num rating",
    "0": "pepper",
    "NaN": "take class again",
    "0.1": "online class rating",
    "0.2": "male",
    "1": "female",
})

rmp_qual_df = rmp_qual_df.rename(columns={
    'Criminal Justice': "major", 
    'George Mason University': "university", 
    'VA': "state",
    })

rmp_tag_df = rmp_tag_df.rename(columns={
    "0": "Tough grader",
    "0.1": "Good feedback",
    "0.2": "Respected",
    "0.3": "Lots to read",
    "0.4": "Participation matters",
    "1": "Don't skip class or you will not pass",
    "0.5": "Lots of homework",
    "0.6": "Inspirational",
    "0.7": "Pop quizzes!",
    "0.8": "Accessible",
    "0.9": "So many papers",
    "0.10": "Clear grading",
    "0.11": "Hilarious",
    "0.12": "Test heavy",
    "0.13": "Graded by few things",
    "0.14": "Amazing lectures",
    "0.15": "Caring",
    "0.16": "Extra credit",
    "0.17": "Group projects",
    "1.1": "Lecture heavy",
})



# %%
print("average rating col missing value rate: ", rmp_num_df["average rating"].isna().sum() / len(rmp_num_df))
print("average difficulty col missing value rate", rmp_num_df["average difficulty"].isna().sum() / len(rmp_num_df))
# %%
num_rating_df = rmp_num_df["num rating"]
print("number of ratings col missing value rate: ",num_rating_df.isna().sum() / len(rmp_num_df))

num_rating_df = num_rating_df.dropna()

plt.hist(num_rating_df, bins=100)
plt.xlabel("number of ratings")
plt.ylabel("count")
plt.show()

thres = 10
print(f"ratio of num rating >= {thres}: ", len(num_rating_df[num_rating_df >= thres]) / len(num_rating_df))

# temp = num_rating_df.value_counts().sort_index()
# print(temp.head(20))
# print(temp.tail())

# %%
# Q1.

temp = rmp_num_df[rmp_num_df["male"] != rmp_num_df["female"]]
temp = temp[temp["num rating"] >= thres]
male_df = temp[temp["male"] == 1]
female_df = temp[temp["female"] == 1]

fig, axes = plt.subplots(1, 2)
fig.supxlabel("number of ratings")
fig.supylabel("frequency")
axes[0].hist(male_df["num rating"], bins=100)
axes[0].set_title("male")

axes[1].hist(female_df["num rating"], bins=100)
axes[1].set_title("female")
plt.show()

fig, axes = plt.subplots(1, 2)
fig.supxlabel("average rating")
fig.supylabel("frequency")
axes[0].hist(male_df["average rating"], bins=10)
axes[0].set_title("male")

axes[1].hist(female_df["average rating"], bins=10)
axes[1].set_title("female")
plt.show()

print("total number of ratings for male professor: ", male_df["num rating"].sum())
print("total number of ratings for female professor: ", female_df["num rating"].sum())

t_stat, p_val = stats.ttest_ind(male_df["average rating"], female_df["average rating"], equal_var=False)
print("Q1. p-value: ", p_val, "t-statistic: ", t_stat)

u_stat, p_val = stats.mannwhitneyu(male_df["average rating"], female_df["average rating"])
print("Q1. p-value: ", p_val, "U statistic: ", u_stat)

# %%
# Q2.

def variance_ratio(sample1, sample2, axis=0):
    var1 = np.var(sample1, axis=axis)
    var2 = np.var(sample2, axis=axis)
    
    assert not np.allclose(var2, np.zeros_like(var2))
    
    return var1 / var2


samples = ((male_df["average rating"].to_numpy(), female_df["average rating"].to_numpy()))

rslt = stats.permutation_test(samples, variance_ratio, n_resamples=1e+4, vectorized=True)
print("Q2. p-value: ", rslt.pvalue, "variance ratio: ", rslt.statistic)

# levene's test
w_stat, p_val = stats.levene(*samples)
print("Q2. (levene's test) p-vaue: ", p_val, "W statistic: ", w_stat)

# %%
# Q3.

def cohens_d(sample1, sample2):
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    n1, n2 = len(sample1), len(sample2)
    pooled_sd = np.sqrt(((n1 - 1) * np.var(sample1) + (n2 - 1) * np.var(sample2)) / (n1 + n2 - 2))
    
    return abs(mean1 - mean2) / pooled_sd


n_exp = 10000
def bootstrap(sample1, sample2, stat_fn, draw_plot=True):
    bs_estimates = []
    for _ in range(n_exp):
        bs_sample1 = np.random.choice(sample1, len(sample1))
        bs_sample2 = np.random.choice(sample2, len(sample2))
        
        bs_estimates.append(stat_fn(bs_sample1, bs_sample2))

    lower_bound = np.percentile(bs_estimates, q=2.5)
    upper_bound = np.percentile(bs_estimates, q=97.5)

    mean_estimate = np.mean(bs_estimates)
    
    conf_interval = (lower_bound, upper_bound)
        
    if draw_plot:
       plt.hist(bs_estimates, bins=50)
       plt.axvline(lower_bound, color='r', linestyle='dashed', linewidth=1.5, label='2.5% Bound')
       plt.axvline(upper_bound, color='r', linestyle='dashed', linewidth=1.5, label='97.5% Bound')
       plt.axvline(mean_estimate, color='y', linestyle='solid', linewidth=1.5, label='mean estimate')
       
       # Adding labels and title
       plt.xlabel('Effect Sizes from Bootstrapping')
       plt.ylabel('Frequency')
       plt.title('Bootstrapped Effect Size Distribution')
       plt.legend()
       plt.show()
   
    return conf_interval

effect_size = cohens_d(*samples)
conf_interval = bootstrap(*samples, cohens_d)
print(f"Q3. Effect size of gender bias in average rating (magnitude of cohen's d'): \
      {effect_size} with confidence interval {conf_interval}")
      
effect_size = variance_ratio(*samples)
conf_interval = bootstrap(*samples, variance_ratio)
print(f"Q3. Effect size of gender bias in spread of average rating (variance ratio): \
      {effect_size} with confidence interval {conf_interval}")

        
        

    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
