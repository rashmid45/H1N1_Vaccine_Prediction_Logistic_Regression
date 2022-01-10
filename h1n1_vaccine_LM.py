# Loading all the required Libraries and Packages


import os
os.chdir('C:\\Users\\shardul\\Desktop\\Rashmi\\H1N1_LOGISTIC')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency 
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc,roc_auc_score

# load the data
df =pd.read_csv('h1n1_vaccine_prediction.csv')

df.info()
'''
RangeIndex: 26707 entries, 0 to 26706
Data columns (total 34 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   unique_id                  26707 non-null  int64  
 1   h1n1_worry                 26615 non-null  float64
 2   h1n1_awareness             26591 non-null  float64
 3   antiviral_medication       26636 non-null  float64
 4   contact_avoidance          26499 non-null  float64
 5   bought_face_mask           26688 non-null  float64
 6   wash_hands_frequently      26665 non-null  float64
 7   avoid_large_gatherings     26620 non-null  float64
 8   reduced_outside_home_cont  26625 non-null  float64
 9   avoid_touch_face           26579 non-null  float64
 10  dr_recc_h1n1_vacc          24547 non-null  float64
 11  dr_recc_seasonal_vacc      24547 non-null  float64
 12  chronic_medic_condition    25736 non-null  float64
 13  cont_child_undr_6_mnths    25887 non-null  float64
 14  is_health_worker           25903 non-null  float64
 15  has_health_insur           14433 non-null  float64
 16  is_h1n1_vacc_effective     26316 non-null  float64
 17  is_h1n1_risky              26319 non-null  float64
 18  sick_from_h1n1_vacc        26312 non-null  float64
 19  is_seas_vacc_effective     26245 non-null  float64
 20  is_seas_risky              26193 non-null  float64
 21  sick_from_seas_vacc        26170 non-null  float64
 22  age_bracket                26707 non-null  object 
 23  qualification              25300 non-null  object 
 24  race                       26707 non-null  object 
 25  sex                        26707 non-null  object 
 26  income_level               22284 non-null  object 
 27  marital_status             25299 non-null  object 
 28  housing_status             24665 non-null  object 
 29  employment                 25244 non-null  object 
 30  census_msa                 26707 non-null  object 
 31  no_of_adults               26458 non-null  float64
 32  no_of_children             26458 non-null  float64
 33  h1n1_vaccine               26707 non-null  int64  
dtypes: float64(23), int64(2), object(9)
memory usage: 6.9+ MB
'''

df.describe()
df.shape 
# 26707 observations and 34 variables

### First check the response variable - h1n1_vaccine

df.h1n1_vaccine.describe()
df.h1n1_vaccine.value_counts()
'''
df.h1n1_vaccine.value_counts()
Out[16]: 
0    21033
1     5674
Name: h1n1_vaccine, dtype: int64
'''
sns.countplot(df.h1n1_vaccine) # unbalanced data 
sum(df.h1n1_vaccine.value_counts()) # 26707 -- no missing values

 
######### CHECKING ALL CATEGORICAL VARIABLES #########

## Categorical variable 1 - age_bracket

df.age_bracket.value_counts() 
'''
Out[23]: 
65+ Years        6843
55 - 64 Years    5563
45 - 54 Years    5238
18 - 34 Years    5215
35 - 44 Years    3848
Name: age_bracket, dtype: int64
'''
sns.countplot(df.age_bracket)
sum(df.age_bracket.value_counts()) # 26707 -- no missing values

# merging categories so that we would get primarily young , middle age and senior ages

df['age_bracket']=df.get('age_bracket').replace('18 - 34 Years','Age_1')
df['age_bracket']=df.get('age_bracket').replace('35 - 44 Years','Age_1')
df['age_bracket']=df.get('age_bracket').replace('45 - 54 Years','Age_2')
df['age_bracket']=df.get('age_bracket').replace('55 - 64 Years','Age_2')
df['age_bracket']=df.get('age_bracket').replace('65+ Years','Age_3')

df.age_bracket.value_counts()
sns.countplot(df.age_bracket)

pd.crosstab(df.age_bracket, df.h1n1_vaccine).values
'''
Out[27]: 
array([[7311, 1752],
       [8430, 2371],
       [5292, 1551]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.age_bracket, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value 1.58e-07 ; <0.05 ; reject Ho; good predictor


#Categorical Variable 2 - qualification

df.qualification.value_counts()
'''
df.qualification.value_counts()
Out[28]: 
College Graduate    10097
Some College         7043
12 Years             5797
< 12 Years           2363
Name: qualification, dtype: int64
'''

sns.countplot(df.qualification) 
sum(df.qualification.value_counts())# 25300 -- 5 % missing values
df.qualification.isnull().sum() # 1407

# imputing missing values by mode
df.qualification.fillna('College Graduate', inplace = True)
sum(df.qualification.value_counts()) # 26707

sns.countplot(df.qualification)

#merging the <12 , 12 and some college into  non-graduate categories

df['qualification']=df.get('qualification').replace('Some College','Non_graduate')
df['qualification']=df.get('qualification').replace('12 Years','Non_graduate')
df['qualification']=df.get('qualification').replace('< 12 Years','Non_graduate')

df.qualification.value_counts()
sns.countplot(df.qualification)

pd.crosstab(df.qualification, df.h1n1_vaccine).values
'''
Out[44]: 
array([[ 8760,  2744],
       [12273,  2930]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.qualification, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 1.48e-19 ; reject Ho ; good predictor


## Categorical variable 3 - race

df.race.value_counts()
'''
df.race.value_counts()
Out[32]: 
White                21222
Black                 2118
Hispanic              1755
Other or Multiple     1612
Name: race, dtype: int64
'''

sns.countplot(df.race)
sum(df.race.value_counts()) # 26707 -- no missing values

#merging the Black,Hispanic and other into non-white categories

df['race']=df.get('race').replace('Black','Non_White')
df['race']=df.get('race').replace('Hispanic','Non_White')
df['race']=df.get('race').replace('Other or Multiple','Non_White')

df.race.value_counts()
sns.countplot(df.race)
(5485/26707)*100 #20%

# biased variable ; not a good predictor


## categorical variable 4 - sex

df.sex.value_counts()
'''
df.sex.value_counts()
Out[89]: 
Female    15858
Male      10849
Name: sex, dtype: int64
'''

sns.countplot(df.sex)
sum(df.sex.value_counts()) # 26707 -- no missing values

pd.crosstab(df.sex, df.h1n1_vaccine).values
'''
Out[58]: 
array([[12378,  3480],
       [ 8655,  2194]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.sex, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 0.00077 i.e < 0.05 ; reject HO ; good predictor


## Categorical variable 5 - income_level

df.income_level.value_counts()
'''
df.income_level.value_counts()
Out[95]: 
<= $75,000, Above Poverty    12777
> $75,000                     6810
Below Poverty                 2697
Name: income_level, dtype: int6
'''

sns.countplot(df.income_level)
sum(df.income_level.value_counts()) # 22284 -- 16%  missing values
df.income_level.isna().sum() #  4423 missing values

# imputing missing values by mode
df.income_level.fillna('<= $75,000, Above Poverty', inplace = True)
sum(df.income_level.value_counts()) # 26707

df.income_level.value_counts()
sns.countplot(df.income_level)

pd.crosstab(df.income_level, df.h1n1_vaccine).values
'''
Out[69]: 
array([[13765,  3435],
       [ 5087,  1723],
       [ 2181,   516]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.income_level, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 1.88e-20 < 0.05 ; reject Ho;  good predictor


## Categorical variable 6 - marital_status

df.marital_status.value_counts()
'''
df.marital_status.value_counts()
Out[98]: 
Married        13555
Not Married    11744
Name: marital_status, dtype: int64
'''

sns.countplot(df.marital_status)
sum(df.marital_status.value_counts()) # 25299 -- 5%  missing values
df.marital_status.isna().sum()# 1408

# imputing missing values by mode
df.marital_status.fillna('Married', inplace = True)
sum(df.marital_status.value_counts()) # 26707

df.marital_status.value_counts()
sns.countplot(df.marital_status)

pd.crosstab(df.marital_status, df.h1n1_vaccine).values
'''
Out[79]: 
array([[11539,  3424],
       [ 9494,  2250]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.marital_status, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 1.69e-13 < 0.05 ; reject Ho; good predictor


## Categorical variable 7 - housing_status

df.housing_status.value_counts()
'''
df.housing_status.value_counts()
Out[116]: 
Own     18736
Rent     5929
Name: housing_status, dtype: int64
'''

sns.countplot(df.housing_status)
sum(df.housing_status.value_counts()) # 24665 -- 7%  missing values
df.housing_status.isna().sum() # 2042 missing values

# imputing missing values by mode
df.housing_status.fillna('Own', inplace = True)
sum(df.housing_status.value_counts()) # 26707

df.housing_status.value_counts()
sns.countplot(df.housing_status)
(5929/20778)*100
pd.crosstab(df.housing_status, df.h1n1_vaccine).values
'''
Out[90]: 
array([[16223,  4555],
       [ 4810,  1119]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.housing_status, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 4.55e-07 < 0.05 ; reject HO; good predictor


## Categorical variable 8 - employment

df.employment.value_counts()
'''
Out[127]: 
Employed              13560
Not in Labor Force    10231
Unemployed             1453
Name: employment, dtype: int64

'''

sns.countplot(df.employment)
sum(df.employment.value_counts()) # 25244 --  8 % missing values

# imputing missing values by mode
df.employment.fillna('Employed', inplace = True)
sum(df.employment.value_counts()) # 26707

df.employment.value_counts()
sns.countplot(df.employment)

pd.crosstab(df.employment, df.h1n1_vaccine).values
'''
Out[99]: 
array([[11829,  3194],
       [ 7988,  2243],
       [ 1216,   237]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.employment, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 6.27e-06 < 0.05  ; Reject Ho ; good predictor


## Categorical variable 9 - census_msa

df.census_msa.value_counts()
'''
df.census_msa.value_counts()
Out[138]: 
MSA, Not Principle  City    11645
MSA, Principle City          7864
Non-MSA                      7198
Name: census_msa, dtype: int64
'''

sns.countplot(df.census_msa)
sum(df.census_msa.value_counts()) # 26707 --  no missing values

#merging the both MSA principle and Not Principle city in single category MSA

df['census_msa']=df.get('census_msa').replace('MSA, Not Principle  City','MSA')
df['census_msa']=df.get('census_msa').replace('MSA, Principle City','MSA')

sns.countplot(df.census_msa)
df.census_msa.value_counts()

pd.crosstab(df.census_msa, df.h1n1_vaccine).values
'''
Out[113]: 
array([[15361,  4148],
       [ 5672,  1526]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.census_msa, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 0.92 > 0.05 ; accept HO;  not a good predictor



## Categorical variable 13 - contact_avoidance

df.contact_avoidance.value_counts()
'''
df.contact_avoidance.value_counts()
Out[130]: 
1.0    19228
0.0     7271
Name: contact_avoidance, dtype: int64
'''

sns.countplot(df.contact_avoidance)
sum(df.contact_avoidance.value_counts()) # 26499 -- almost 1% missing values
df.contact_avoidance.isna().sum() # 208 missing values

# imputing missing values by mode
df.contact_avoidance.fillna(1.0, inplace = True)
sum(df.contact_avoidance.value_counts()) # 26707

df.contact_avoidance.isna().sum()
df.contact_avoidance.value_counts()
sns.countplot(df.contact_avoidance)

pd.crosstab(df.contact_avoidance, df.h1n1_vaccine).values
'''
Out[163]: 
array([[ 5954,  1317],
       [15079,  4357]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.contact_avoidance, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 2.21e014 < 0.05 ; reject HO;  a good predictor



## Categorical variable 14 - bought_face_mask

df.bought_face_mask.value_counts()
'''
df.bought_face_mask.value_counts()
Out[149]: 
0.0    24847
1.0     1841
Name: bought_face_mask, dtype: int64
'''

sns.countplot(df.bought_face_mask)
sum(df.bought_face_mask.value_counts()) # 26688 -- very few missing values
df.bought_face_mask.isna().sum() # 19 

# imputing missing values by mode
df.bought_face_mask.fillna(0.0, inplace = True)
sum(df.bought_face_mask.value_counts()) # 26707

df.bought_face_mask.value_counts()
((1841)/26707)*100 # only 7% 

# This variable is biased by category '0'  ; not a good predictor



## Categorical variable 15 - wash_hands_frequently

df.wash_hands_frequently.value_counts()
'''
df.wash_hands_frequently.value_counts()
Out[159]: 
1.0    22015
0.0     4650
Name: wash_hands_frequently, dtype: int64
'''
sns.countplot(df.wash_hands_frequently)
sum(df.wash_hands_frequently.value_counts()) # 26665 -- very few missing values
df.wash_hands_frequently.isna().sum() # 42 missing values

# imputing missing values by mode
df.wash_hands_frequently.fillna(1.0, inplace = True)
sum(df.wash_hands_frequently.value_counts()) # 26707

df.wash_hands_frequently.value_counts()
sns.countplot(df.wash_hands_frequently)
((4650)/26707)*100 # 17%

#This variable is biased by category '1' hence not a good predictor


## Categorical variable 16 - avoid_large_gatherings

df.avoid_large_gatherings.value_counts()
'''
df.avoid_large_gatherings.value_counts()
Out[170]: 
0.0    17073
1.0     9547
Name: avoid_large_gatherings, dtype: int644
'''
sns.countplot(df.avoid_large_gatherings)
sum(df.avoid_large_gatherings.value_counts()) # 26620 -- very few missing values
df.avoid_large_gatherings.isna().sum() # 87

# imputing missing values by mode
df.avoid_large_gatherings.fillna(0.0, inplace = True)
sum(df.avoid_large_gatherings.value_counts()) # 26707

pd.crosstab(df.avoid_large_gatherings, df.h1n1_vaccine).values
'''
Out[186]: 
array([[13609,  3551],
       [ 7424,  2123]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.avoid_large_gatherings, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value 0.0032 < 0.05 ; reject HO ; good predictor


## Categorical variable 17 - avoid_touch_face

df.avoid_touch_face.value_counts()
'''
df.avoid_touch_face.value_counts()
Out[205]: 
1.0    18001
0.0     8578
Name: avoid_touch_face, dtype: int64
'''
sns.countplot(df.avoid_touch_face)
sum(df.avoid_touch_face.value_counts()) # 26579 -- very few missing values

# imputing missing values by mode
df.avoid_touch_face.fillna(1.0, inplace = True)
sum(df.avoid_touch_face.value_counts()) # 26707

pd.crosstab(df.avoid_touch_face, df.h1n1_vaccine).values
'''
Out[193]: 
array([[ 7117,  1461],
       [13916,  4213]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.avoid_touch_face, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 6.32e-31 < 0.05 ; reject HO ; good predictor


## Categorical variable 18 - reduced_outside_home_cont

df.reduced_outside_home_cont.value_counts()
'''
df.reduced_outside_home_cont.value_counts()
Out[227]: 
0.0    17644
1.0     8981
Name: reduced_outside_home_cont, dtype: int64
'''
sns.countplot(df.reduced_outside_home_cont)
sum(df.reduced_outside_home_cont.value_counts()) # 26625 -- very few missing values
df.reduced_outside_home_cont.isna().sum() #82 missing values

# imputing missing values by mode
df.reduced_outside_home_cont.fillna(0.0, inplace = True)
sum(df.reduced_outside_home_cont.value_counts()) # 26707

pd.crosstab(df.reduced_outside_home_cont, df.h1n1_vaccine).values
'''
Out[202]: 
array([[14074,  3652],
       [ 6959,  2022]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.reduced_outside_home_cont, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value 0.00032 < 0.05 ; reject HO ; good predictor


## Categorical variable 19 - dr_recc_h1n1_vacc

df.dr_recc_h1n1_vacc.value_counts()
'''
df.dr_recc_h1n1_vacc.value_counts()
Out[232]: 
0.0    19139
1.0     5408
Name: dr_recc_h1n1_vacc, dtype: int64
'''
sns.countplot(df.dr_recc_h1n1_vacc)
sum(df.dr_recc_h1n1_vacc.value_counts()) # 24547 -- 8% few missing values
df.dr_recc_h1n1_vacc.isna().sum() # 2160 missing values

# imputing missing values by mode
df.dr_recc_h1n1_vacc.fillna(0.0, inplace = True)
sum(df.dr_recc_h1n1_vacc.value_counts()) # 26707
df.dr_recc_h1n1_vacc.value_counts()
(5408/26707)*100 

pd.crosstab(df.dr_recc_h1n1_vacc, df.h1n1_vaccine).values
'''
Out[214]: 
array([[18504,  2795],
       [ 2529,  2879]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.dr_recc_h1n1_vacc, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value is 0.0 < 0.05 ; reject HO ; good predictor


## Categorical variable 20 - dr_recc_seasonal_vacc

df.dr_recc_seasonal_vacc.value_counts()
'''
df.dr_recc_seasonal_vacc.value_counts()
Out[247]: 
0.0    16453
1.0     8094
Name: dr_recc_seasonal_vacc, dtype: int64
'''
sns.countplot(df.dr_recc_seasonal_vacc)
sum(df.dr_recc_seasonal_vacc.value_counts()) # 24547 -- 8% few missing values
df.dr_recc_seasonal_vacc.isna().sum() # 2160 missing values

# imputing missing values by mode
df.dr_recc_seasonal_vacc.fillna(0.0, inplace = True)
sum(df.dr_recc_seasonal_vacc.value_counts()) # 26707

pd.crosstab(df.dr_recc_seasonal_vacc, df.h1n1_vaccine).values
'''
Out[224]: 
array([[15758,  2855],
       [ 5275,  2819]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.dr_recc_seasonal_vacc, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value 3.31e-280 < 0.05 ; reject HO ; good predictor


## Categorical variable 21 - chronic_medic_condition

df.chronic_medic_condition.value_counts()
'''
df.chronic_medic_condition.value_counts()
Out[260]: 
0.0    18446
1.0     7290
Name: chronic_medic_condition, dtype: int64
'''
sns.countplot(df.chronic_medic_condition)
sum(df.chronic_medic_condition.value_counts()) # 25736 -- 4% few missing values

# imputing missing values by mode
df.chronic_medic_condition.fillna(0.0, inplace = True)
sum(df.chronic_medic_condition.value_counts()) # 26707

pd.crosstab(df.chronic_medic_condition, df.h1n1_vaccine).values
'''
Out[232]: 
array([[15751,  3666],
       [ 5282,  2008]], dtype=int64)
'''

#plotting chi sq test
chi2, p, dof, expected = chi2_contingency((pd.crosstab(df.chronic_medic_condition, df.h1n1_vaccine).values))
print (f'Chi-square Statistic : {chi2} ,p-value: {p}')
# p value 1.542e-53 < 0.05 ; reject HO ; good predictor


## Categorical variable 22 - cont_child_undr_6_mnths

df.cont_child_undr_6_mnths.value_counts()
'''
df.cont_child_undr_6_mnths.value_counts()
Out[274]: 
0.0    23749
1.0     2138
Name: cont_child_undr_6_mnths, dtype: int64
'''
sns.countplot(df.cont_child_undr_6_mnths)
sum(df.cont_child_undr_6_mnths.value_counts()) # 25887 -- 3 % few missing values
df.cont_child_undr_6_mnths.isna().sum()  # 820 missing values

# imputing missing values by mode
df.cont_child_undr_6_mnths.fillna(0.0, inplace = True)
sum(df.cont_child_undr_6_mnths.value_counts()) # 26707

df.cont_child_undr_6_mnths.value_counts()
sns.countplot(df.cont_child_undr_6_mnths)
(2138/26707)*100  # only 8% 

# biased variable by category 0 ; not a good predictor



## Categorical variable 23 - is_health_worker

df.is_health_worker.value_counts()
'''
df.is_health_worker.value_counts()
Out[282]: 
0.0    23004
1.0     2899
Name: is_health_worker, dtype: int64
'''
sns.countplot(df.is_health_worker)
sum(df.is_health_worker.value_counts()) # 25903 -- 3 % few missing values
df.is_health_worker.isna().sum() # 804 missing values

# imputing missing values by mode
df.is_health_worker.fillna(0.0, inplace = True)
sum(df.is_health_worker.value_counts()) # 26707

df.is_health_worker.value_counts()
sns.countplot(df.is_health_worker)
(2899/26707)*100 
# biased variable in 90:10 ratio; not a good predictor



## Categorical variable 24 - has_health_insur

df.has_health_insur.value_counts()
'''
df.has_health_insur.value_counts()
Out[288]: 
1.0    12697
0.0     1736
Name: has_health_insur, dtype: int64
'''
sns.countplot(df.has_health_insur)
sum(df.has_health_insur.value_counts()) # 14433 -- 46 % few missing values
df.has_health_insur.isna().sum() # 12274 missing  values

# imputing missing values by mode
df.has_health_insur.fillna(1.0, inplace = True)
sum(df.has_health_insur.value_counts()) # 26707

df.has_health_insur.value_counts()
sns.countplot(df.has_health_insur)
(1736/26707)*100 # only 7%
# biased variable ;not a good predictor


######### CHECKING ORDINAL CATEGORICAL VARIABLE ##########

## Ordinal Categorical variable 1 - no_of_adults

df.no_of_adults.describe() # count 26458
'''
df.no_of_adults.describe()
Out[31]: 
count    26458.000000
mean         0.886499
std          0.753422
min          0.000000
25%          0.000000
50%          1.000000
75%          1.000000
max          3.000000
Name: no_of_adults, dtype: float64
'''

# histogram
plt.hist(df.no_of_adults, bins = 'auto' , facecolor = 'b')
plt.xlabel('no_of_adults')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['no_of_adults'].plot.box(patch_artist = True, vert = False)
# we can tolerate this outlier

df.no_of_adults.isna().sum() # 249 missing values

# imputing missing values by mode
df.no_of_adults.fillna(df.no_of_adults.mean(), inplace = True)
sum(df.no_of_adults.value_counts()) # 26707


# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.no_of_adults, df_1.no_of_adults)
# p value is 0.21 > 0.05 ; accept Ho; not a good predictor



## Ordinal Categorical variable 2 - no_of_children

df.no_of_children.describe()
'''
Out[43]: 
count    26458.000000
mean         0.534583
std          0.928173
min          0.000000
25%          0.000000
50%          0.000000
75%          1.000000
max          3.000000
Name: no_of_children, dtype: float64
'''

# histogram
plt.hist(df.no_of_children, bins = 'auto' , facecolor = 'b')
plt.xlabel('no_of_children')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['no_of_children'].plot.box(patch_artist = True, vert = False)
# we can tolerate this outlier

df.no_of_children.isna().sum() # 249 missing values

# imputing missing values by mode
df.no_of_children.fillna(df.no_of_children.mean(), inplace = True)
sum(df.no_of_children.value_counts()) # 26707


# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.no_of_children, df_1.no_of_children)
# p value is 0.588 > 0.05 ; accept  HO; not a good predictor



## Ordinal Categorical variable 3 - antiviral_medication

df.antiviral_medication.value_counts()

'''
df.antiviral_medication.value_counts()
Out[151]: 
0.0    25335
1.0     1301
Name: antiviral_medication, dtype: int64
'''

sns.countplot(df.antiviral_medication)
sum(df.antiviral_medication.value_counts()) # 26636 --  2% missing values
df.antiviral_medication.isna().sum() # 71 missing values

sns.countplot(df.antiviral_medication)
((1301)/26707)*100 # only 5%

# Even after imputing 2% missing values , its obvious that this variable is biased by category '0' ; not a good predictor




## Ordinal Categorical 4 -  is_h1n1_vacc_effective

df.is_h1n1_vacc_effective.describe()  # count 26316
df.is_h1n1_vacc_effective.isna().sum() # 391 missing values 
'''
df.is_h1n1_vacc_effective.describe() 
Out[25]: 
count    26316.000000
mean         3.850623
std          1.007436
min          1.000000
25%          3.000000
50%          4.000000
75%          5.000000
max          5.000000
Name: is_h1n1_vacc_effective, dtype: float64
'''

# histogram
plt.hist(df.is_h1n1_vacc_effective, bins = 'auto' , facecolor = 'b')
plt.xlabel('is_h1n1_vacc_effective')
plt.ylabel('Frequency')

# boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['is_h1n1_vacc_effective'].plot.box(patch_artist = True, vert = False)

sns.boxplot(df.is_h1n1_vacc_effective) # no outliers


# replace missing with mean (or any number)
df['is_h1n1_vacc_effective'] = df['is_h1n1_vacc_effective'].fillna(df['is_h1n1_vacc_effective'].median())
df.is_h1n1_vacc_effective.describe() # count = 26707


# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.is_h1n1_vacc_effective, df_1.is_h1n1_vacc_effective)
#p value is 0.0  < 0.05; reject Ho ; good predictor



## Ordinal Categorical 5 -  is_h1n1_risky

df.is_h1n1_risky.describe() 
'''
Out[109]: 
count    26319.000000
mean         2.342566
std          1.285539
min          1.000000
25%          1.000000
50%          2.000000
75%          4.000000
max          5.000000
Name: is_h1n1_risky, dtype: float64
'''

# histogram
plt.hist(df.is_h1n1_risky, bins = 'auto' , facecolor = 'b')
plt.xlabel('is_h1n1_risky')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['is_h1n1_risky'].plot.box(patch_artist = True, vert = False)
sns.boxplot(df.is_h1n1_vacc_effective) # no outliers

df['is_h1n1_risky'].value_counts()

# replace missing with mean (or any number)
df['is_h1n1_risky'] = df['is_h1n1_risky'].fillna(df['is_h1n1_risky'].median())
df.is_h1n1_risky.describe() # count = 26707

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.is_h1n1_risky, df_1.is_h1n1_risky,nan_policy='omit')
#p value is 0.0  < 0.05; reject Ho ; good predictor



## Ordinal Categorical 6 -  sick_from_h1n1_vacc

df.sick_from_h1n1_vacc.describe() 
df.sick_from_h1n1_vacc.isna().sum() # 395 missing values 
'''
df.sick_from_h1n1_vacc.describe() 
Out[40]: 
count    26312.000000
mean         2.357670
std          1.362766
min          1.000000
25%          1.000000
50%          2.000000
75%          4.000000
max          5.000000
Name: sick_from_h1n1_vacc, dtype: float64
'''

# histogram
plt.hist(df.sick_from_h1n1_vacc, bins = 'auto' , facecolor = 'b')
plt.xlabel('sick_from_h1n1_vacc')
plt.ylabel('Frequency')


#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['sick_from_h1n1_vacc'].plot.box(patch_artist = True, vert = False)
sns.boxplot(df.sick_from_h1n1_vacc) # no outliers

# replace missing with mean (or any number)
df['sick_from_h1n1_vacc'] = df['sick_from_h1n1_vacc'].fillna(df['sick_from_h1n1_vacc'].median())
df.sick_from_h1n1_vacc.describe() # count = 26707

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.sick_from_h1n1_vacc, df_1.sick_from_h1n1_vacc)
#p value is 0.0  < 0.05; reject Ho ; good predictor



## Ordinal Categorical 7 -  is_seas_risky

df.is_seas_risky.describe() 
df.is_seas_risky.isna().sum() # 514 missing values 
'''
df.is_seas_risky.describe() 
Out[48]: 
count    26193.000000
mean         2.719162
std          1.385055
min          1.000000
25%          2.000000
50%          2.000000
75%          4.000000
max          5.000000
Name: is_seas_risky, dtype: float64
'''

# histogram
plt.hist(df.is_seas_risky, bins = 'auto' , facecolor = 'b')
plt.xlabel('is_seas_risky')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['is_seas_risky'].plot.box(patch_artist = True, vert = False)
sns.boxplot(df.is_seas_risky) # no outliers
sns.countplot(df.is_seas_risky)

# replace missing with mean (or any number)
df['is_seas_risky'] = df['is_seas_risky'].fillna(df['is_seas_risky'].median())
df.is_seas_risky.describe() # count = 26707

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.is_seas_risky, df_1.is_seas_risky)
#p value is 0.0  < 0.05; reject Ho ; good predictor


## Ordinal Categorical 8 -  sick_from_seas_vacc

df.sick_from_seas_vacc.describe()
'''
Out[70]: 
count    26170.000000
mean         2.118112
std          1.332950
min          1.000000
25%          1.000000
50%          2.000000
75%          4.000000
max          5.000000
Name: sick_from_seas_vacc, dtype: float64
'''

df.sick_from_seas_vacc.isna().sum() # 537 missing values 
sns.countplot(df.sick_from_seas_vacc)

# histogram
plt.hist(df.sick_from_seas_vacc, bins = 'auto' , facecolor = 'b')
plt.xlabel('sick_from_seas_vacc')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['sick_from_seas_vacc'].plot.box(patch_artist = True, vert = False)
sns.boxplot(df.is_seas_risky) # no outliers

#imputing missing values
df.sick_from_seas_vacc.fillna(df.sick_from_seas_vacc.median(),inplace=True)
df.sick_from_seas_vacc.describe()

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.sick_from_seas_vacc, df_1.sick_from_seas_vacc)
#p value is 0.16  > 0.05; accept Ho ; not a good predictor


## Ordinal Categorical 9 -  is_seas_vacc_effective

df.is_seas_vacc_effective.describe() 
'''
Out[81]: 
count    26245.000000
mean         4.025986
std          1.086565
min          1.000000
25%          4.000000
50%          4.000000
75%          5.000000
max          5.000000
Name: is_seas_vacc_effective, dtype: float64
'''
# histogram
plt.hist(df.is_seas_vacc_effective, bins = 'auto' , facecolor = 'b')
plt.xlabel('is_seas_vacc_effective')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['is_seas_vacc_effective'].plot.box(patch_artist = True, vert = False)
sns.boxplot(df.is_seas_vacc_effective) # we can tolerate these outliers

#imputing missing values
df.is_seas_vacc_effective.fillna(df.is_seas_vacc_effective.median(),inplace=True)
df.is_seas_vacc_effective.describe()

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.is_seas_vacc_effective, df_1.is_seas_vacc_effective)
#p value is 1.448e-188  < 0.05; reject Ho ; good predictor


## Ordinal Categorical 10 - h1n1_worry

df.h1n1_worry.describe() # count 26615 
df.h1n1_worry.isna().sum()# 92 missing values
'''
df.h1n1_worry.describe()
Out[12]: 
count    26615.000000
mean         1.618486
std          0.910311
min          0.000000
25%          1.000000
50%          2.000000
75%          2.000000
max          3.000000
Name: h1n1_worry, dtype: float64
'''

# histogram
plt.hist(df.h1n1_worry, bins = 'auto' , facecolor = 'b')
plt.xlabel('h1n1_worry')
plt.ylabel('Frequency')

#boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['h1n1_worry'].plot.box(patch_artist = True, vert = False)

sns.boxplot(df.h1n1_worry) # no outliers

# replace missing with mean (or any number)
df['h1n1_worry'] = df['h1n1_worry'].fillna(df['h1n1_worry'].mean())
df.h1n1_worry.describe() # count = 26707

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.h1n1_worry, df_1.h1n1_worry)
# p value is nan < 0.05 ; reject Ho; good predictor



## Ordinal Categorical 10 -  h1n1_awareness

df.h1n1_awareness.describe() # count 26591
df.h1n1_awareness.isna().sum() # 116 missing values 
'''
df.h1n1_awareness.describe() #no missing values
Out[6]: 
count    26591.000000
mean         1.262532
std          0.618149
min          0.000000
25%          1.000000
50%          1.000000
75%          2.000000
max          2.000000
Name: h1n1_awareness, dtype: float64
'''

# histogram
plt.hist(df.h1n1_awareness, bins = 'auto' , facecolor = 'b')
plt.xlabel('h1n1_awareness')
plt.ylabel('Frequency')

# boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['h1n1_awareness'].plot.box(patch_artist = True, vert = False)

sns.boxplot(df.h1n1_awareness) # no outliers

# replace missing with mean 
df['h1n1_awareness'] = df['h1n1_awareness'].fillna(df['h1n1_awareness'].mean())
df.h1n1_awareness.describe() # count = 26707

# Indpndnt sample t test
df_0 = df[df.h1n1_vaccine == 0]
df_1 = df[df.h1n1_vaccine == 1]

stats.ttest_ind(df_0.h1n1_awareness, df_1.h1n1_awareness)
# p value is 5.024e-83 < 0.05 ; reject Ho; good predictor


# copy this dataset to local 
df.to_csv('final_df.csv')





######## EDA is done ; start building the model now #######

df1 =pd.read_csv('final_df.csv')

model1 =smf.glm(formula ='''h1n1_vaccine~h1n1_worry + h1n1_awareness+contact_avoidance+
avoid_large_gatherings+reduced_outside_home_cont+avoid_touch_face+dr_recc_h1n1_vacc+
dr_recc_seasonal_vacc+chronic_medic_condition+is_h1n1_vacc_effective+is_h1n1_risky+
sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+age_bracket+qualification+sex+income_level+marital_status+housing_status+
employment''',data = df1, family =sm.families.Binomial())

result = model1.fit()
print(result.summary())

## plotting model again -- removed income_level and employment based on pvalue

model2 =smf.glm(formula ='''h1n1_vaccine~h1n1_worry + h1n1_awareness+
avoid_large_gatherings+avoid_touch_face+dr_recc_h1n1_vacc+
dr_recc_seasonal_vacc+chronic_medic_condition+is_h1n1_vacc_effective+is_h1n1_risky+
sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+age_bracket+qualification+sex+marital_status+housing_status''',data = df1, family =sm.families.Binomial())

result2 = model2.fit()
print(result2.summary())

#filtering out variables based on p value again 
# removed sex,avoid_touch_face and housing_status and chronic_medic_condition 


model3 =smf.glm(formula ='''h1n1_vaccine~h1n1_worry + h1n1_awareness+
avoid_large_gatherings+dr_recc_h1n1_vacc+
dr_recc_seasonal_vacc+is_h1n1_vacc_effective+is_h1n1_risky+
sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+age_bracket+qualification
+marital_status''',data = df1, family =sm.families.Binomial())

result3 = model3.fit()
print(result3.summary())

predictions = result3.predict()
predictions
predictions_nominal  = [ 0 if x < 0.5 else 1 for x in predictions]


# confusion Matrix


print(confusion_matrix (df1['h1n1_vaccine'], predictions_nominal))
'''
[[19957  1076]
 [ 3430  2244]]
'''

# Accuracy Score
(19957+2244)/(19957+2244+1076+3430) # 83%
# ROC and  AUC

fpr,tpr,thresholds = roc_curve(df1['h1n1_vaccine'], predictions)
roc_auc = auc(fpr,tpr)
print(roc_auc) # 82.3%

#ROC curve
plt.title('ROC curve for H1N1 vaccine classifier')
plt.plot([0,1],[0,1],'b--')
plt.plot(fpr,tpr,label ='AUC = ' +str(roc_auc))
plt.xlabel('flase postitive rate (1-specificty)')
plt.ylabel('true positive rate (sensitiviy)')


# classification report
print(classification_report(df1['h1n1_vaccine'],predictions_nominal,digits =3))
'''
              precision    recall  f1-score   support

           0      0.853     0.949     0.899     21033
           1      0.676     0.395     0.499      5674

    accuracy                          0.831     26707
   macro avg      0.765     0.672     0.699     26707
weighted avg      0.816     0.831     0.814     26707
'''

## VIF

h1n1_worry + h1n1_awareness+
avoid_large_gatherings+dr_recc_h1n1_vacc+
dr_recc_seasonal_vacc+is_h1n1_vacc_effective+is_h1n1_risky+
sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+
age_bracket+qualification
+marital_status

df1.info()
df2 = df1.iloc[:,[2,3,8,11,12,17,18,19,20,21,23,24,28]]

df2.info()

le = LabelEncoder()

df2.iloc[:,10:] = df2.iloc[:,10:].apply(le.fit_transform)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
df2.columns
vif_data['feature']=df2.columns
vif_data.head()

len(df2.columns)
range(len(df2.columns))

vif_data['VIF'] = [variance_inflation_factor(df2.values , i)
                   for i in range(len(df2.columns))]

print(vif_data)
'''
print(vif_data)
                   feature        VIF
0               h1n1_worry   5.693184
1           h1n1_awareness   4.860759
2   avoid_large_gatherings   1.729302
3        dr_recc_h1n1_vacc   2.083193
4    dr_recc_seasonal_vacc   2.404706
5   is_h1n1_vacc_effective  17.882310
6            is_h1n1_risky   7.319892
7      sick_from_h1n1_vacc   4.752438
8   is_seas_vacc_effective  18.978029
9            is_seas_risky   8.016106
10             age_bracket   2.542390
11           qualification   2.396817
12          marital_status   1.816581
'''


vif_data[vif_data['VIF']>=10]
'''
Out[199]: 
                  feature        VIF
5  is_h1n1_vacc_effective  17.882310
8  is_seas_vacc_effective  18.978029

'''
# since is_h1n1_vacc_effective and is_seas_vacc_effective ; we will not consider is_seas_vacc_effective
# and proceed to bulding a model again after validating VIF

model4 =smf.glm(formula ='''h1n1_vaccine~h1n1_worry + h1n1_awareness+
avoid_large_gatherings+dr_recc_h1n1_vacc+
dr_recc_seasonal_vacc+is_h1n1_vacc_effective+is_h1n1_risky+
sick_from_h1n1_vacc+is_seas_risky+age_bracket+qualification
+marital_status''',data = df1, family =sm.families.Binomial())

result4 = model4.fit()
print(result4.summary())

predictions = result4.predict()
predictions
predictions_nominal  = [ 0 if x < 0.5 else 1 for x in predictions]

# confusion Matrix
print(confusion_matrix (df1['h1n1_vaccine'], predictions_nominal))
'''
[[19954  1079]
 [ 3442  2232]]
'''

# Accuracy Score
(19954+2232)/(19954+2232+1079+3442) # 83%
# ROC and  AUC

fpr,tpr,thresholds = roc_curve(df1['h1n1_vaccine'], predictions)
roc_auc = auc(fpr,tpr)
print(roc_auc) # 82.3%

#ROC curve
plt.title('ROC curve for H1N1 vaccine classifier - after VIF')
plt.plot([0,1],[0,1],'b--')
plt.plot(fpr,tpr,label ='AUC = ' +str(roc_auc))
plt.xlabel('flase postitive rate (1-specificty)')
plt.ylabel('true positive rate (sensitiviy)')


# classification report
print(classification_report(df1['h1n1_vaccine'],predictions_nominal,digits =3))

