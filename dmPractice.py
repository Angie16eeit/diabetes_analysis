#Diabetes practice

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "/Users/angiehsin/VsCodeProjects//diabetes_analysis/archive"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#####[1]
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot

import re
import sklearn

import warnings
warnings.filterwarnings("ignore")

plt.show()


# 設定檔案路徑來源
file_path = '../diabetes_analysis/archive/'

# 輸出檔案路徑來源
output_path = '/Users/angiehsin/Desktop/figure_dm/'

# 讀取CSV文件
df1 = pd.read_csv(file_path + 'labs.csv')
df2 = pd.read_csv(file_path + 'examination.csv')
df3 = pd.read_csv(file_path + 'demographic.csv')
df4 = pd.read_csv(file_path + 'diet.csv')
df5 = pd.read_csv(file_path + 'questionnaire.csv')
df6 = pd.read_csv(file_path + 'medications.csv', encoding='latin1')

df2.drop(['SEQN'], axis = 1, inplace=True)
df3.drop(['SEQN'], axis = 1, inplace=True)
df4.drop(['SEQN'], axis = 1, inplace=True)
df5.drop(['SEQN'], axis = 1, inplace=True)
df6.drop(['SEQN'], axis=1, inplace=True)

## [1] 合併表格 , 匯出所有資料
df = pd.concat([df1, df2], axis=1, join='inner')
df = pd.concat([df, df3], axis=1, join='inner')
df = pd.concat([df, df4], axis=1, join='inner')
df = pd.concat([df, df5], axis=1, join='inner')
df = pd.concat([df, df6], axis=1, join='inner')

#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#sel.fit_transform(df)

#df.describe()
print('-----[1] 合併表格 , 匯出所有資料-----')
all_data = df.describe()
all_data.to_csv(output_path + '01_all_data.csv')
print(df.describe())

##------ [2] 提取需要表格內容，匯出資料
from sklearn.feature_selection import VarianceThreshold

df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')

df = df.rename(columns = {'SEQN' : 'ID',
                          'RIAGENDR' : 'Gender',
                          'PEASCST1' : 'BP_Status',
                          'LBXGH' : 'Glycohemoglobin',
                          'BMDAVSAD' : 'Sagittal_Abdominal_Diameter',
                          'DMDHRAGE': 'Age',
                          'BMXBMI' : 'BMI',
                          'BMXWAIST' : 'Waist_Circum',
                          'BMXWT' : 'Weight_kg',
                          'LBXSGL': 'Glucose_mg',
                          'LBXGLT': 'Two_Hour_Glucose_mg'}) 
# LBXSGL: Glucose, refrigerated serum (mg/dL)
# LBDSGLSI: Glucose, refrigerated serum (mmol/L)
# LBXGLT : Two Hour Glucose(OGTT) (mg/dL)
# LBDGLTSI: Two Hour Glucose(OGTT) (mmol/L)
# DMDHRAGE: HH reference person's age in years

## 補充：測量矢狀腹徑（SAD；Sagittal Abdominal Diameter）了解腹部肥胖指數
df = df.loc[:, ['ID', 'Gender', 'BP_Status', 'Glycohemoglobin', 'Sagittal_Abdominal_Diameter', 'Age', 'BMI', 'Waist_Circum', 'Weight_kg', 'Glucose_mg', 'Two_Hour_Glucose_mg']]

# df = df.loc[:, ['ID', 'Gender', 'BP_Status', 'Glyco_Hemoglobin', 'Sagittal_Abdominal_Diameter', 'Age', 'BMI', 'Waist_Circum', 'Weight_kg', 'Glucose_mg', 'Glucose_mmol', 'Two_Hour_Glucose_mg', 'Two_Hour_Glucose_mmol']]

# df = df.loc[:, ['ID', 'Gender', 'Marital_Status', 'Years_in_US', 'Family_income', 'BMI', 'BP_Status', 'Weight_kg', 'SaggitalAbdominal', 'WaistCircum']]
print('-----[2] 提取所需內容，匯出資料 -----')
filter_data = df
filter_data.to_csv(output_path + '02_filter_data.csv')
filter_data_describe = df.describe()
filter_data_describe.to_csv(output_path + '02_filter_data_describe.csv')
print(df.describe())

##------ [3] 補足欄位null
from sklearn.feature_selection import VarianceThreshold

#year in us -> american : 0, not american : 1
df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')

#GlycoHemoglobin, Saggital Abdominal(median)
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Glycohemoglobin'] = df['Glycohemoglobin'].fillna(df['Glycohemoglobin'].median())
df['Sagittal_Abdominal_Diameter'] = df['Sagittal_Abdominal_Diameter'].fillna(df['Sagittal_Abdominal_Diameter'].median())
df['Waist_Circum'] = df['Waist_Circum'].fillna(df['Waist_Circum'].median())
df['Weight_kg'] = df['Weight_kg'].fillna(df['Weight_kg'].median())
df['Glucose_mg'] = df['Glucose_mg'].fillna(df['Glucose_mg'].median())
# df['Glucose_mmol'] = df['Glucose_mmol'].fillna(df['Glucose_mmol'].median())
df['Two_Hour_Glucose_mg'] = df['Two_Hour_Glucose_mg'].fillna(df['Two_Hour_Glucose_mg'].median())
# df['Two_Hour_Glucose_mmol'] = df['Two_Hour_Glucose_mmol'].fillna(df['Two_Hour_Glucose_mmol'].median())

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(df)

#for dataset in df:
#    dataset['GlycoHemoglobin'] = dataset['GlycoHemoglobin'].fillna(df['GlycoHemoglobin'].median())

#df.head(12)
print('-----[3] 補na，匯出資料 -----')
fill_na_data = df
fill_na_data.to_csv(output_path + '03_fill_na_data.csv')
fill_na_data_describe = df.describe()
fill_na_data_describe.to_csv(output_path + '03_fill_na_data_describe.csv')
print(fill_na_data_describe)

##------ [4] 是否有糖尿病？0=無/1= 高風險 /2 =有DM
df.loc[df['Glycohemoglobin'] < 6.0, 'Diabetes'] = 0
df.loc[(df['Glycohemoglobin'] >= 6.0) & (df['Glycohemoglobin'] <= 6.4), 'Diabetes'] = 1
df.loc[df['Glycohemoglobin'] >= 6.5, 'Diabetes'] = 2

print('-----[4] 是否有糖尿病？0=無/1= 高風險 /2 =有DM -----')
is_dm_data = df
is_dm_data.to_csv(output_path + '04_is_dm_data.csv')
#顯示前10筆
print(df.head(10))

##------ [5] 繪出相關係數圖(熱度圖)
colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
sns.heatmap(df.astype(float).drop(axis=1, labels='ID').corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True)

print('-----[5]繪出相關係數圖，自動存圖-----')
# 保存圖表為PNG文件
plt.savefig(output_path + '05_heatmap.png')
plt.show()

##------ [6] 匯出多變量圖(pairplot)
show = sns.pairplot(df.drop(['ID', 'Glycohemoglobin'], axis=1), hue='Diabetes', size=1.5, diag_kind='kde')

show.set(xticklabels=[])
print('-----[6] 匯出多變量圖(pairplot)，自動存圖-----')
plt.savefig(output_path + '06_pairplot.png')
# plt.show() 

##------ [7]繪出箱型圖(boxplot)
plt.figure(figsize=(15, 10))
sns.boxplot(data=fill_na_data.drop(columns=['ID', 'Gender', 'BP_Status','Diabetes']))  # 排除非數值列
plt.xticks(rotation=45)
plt.title('Boxplot of Filtered Data')
print('-----[7] 繪出箱型圖(boxplot)，自動存圖-----')
plt.savefig(output_path + '07_boxplot_filter_data.png')
# plt.show()

##------ [8]繪出箱型圖(boxplot)，Diabetes vs Glycohemoglobin
print('-----[8] 繪出箱型圖(boxplot)，自動存圖-----')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='Glycohemoglobin', data=df)
plt.title('Boxplot of Glycohemoglobin Grouped by Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Glycohemoglobin')
plt.savefig(output_path + '08_boxplot_glycohemoglobin_by_diabetes.png')
# plt.show()

##------ [9]繪出箱型圖(boxplot)，Diabetes vs Sagittal_Abdominal_Diameter
print('-----[9] 繪出箱型圖(boxplot)，自動存圖-----')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='Sagittal_Abdominal_Diameter', data=df)
plt.title('Boxplot of Sagittal Abdominal Diameter Grouped by Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Sagittal Abdominal Diameter(SAD)')
plt.savefig(output_path + '09_boxplot_Sagittal_Abdominal_Diameter_by_diabetes.png')
# plt.show()

##------ [10]繪出箱型圖(boxplot)，Diabetes vs BMI
print('-----[10] 繪出箱型圖(boxplot)，自動存圖-----')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='BMI', data=df)
plt.title('Boxplot of BMI Grouped by Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('BMI')
plt.savefig(output_path + '10_boxplot_BMI_by_diabetes.png')
# plt.show()

##------ [11]繪出箱型圖(boxplot)，Diabetes vs Glucose
print('-----[11] 繪出箱型圖(boxplot)，自動存圖-----')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='Glucose_mg', data=df)
plt.title('Boxplot of Glucose Grouped by Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Glucose(mg/mL)')
plt.savefig(output_path + '11_boxplot_Glucose_by_diabetes.png')
# plt.show()

##------ [12]繪出箱型圖(boxplot)，Diabetes vs Waist
print('-----[12] 繪出箱型圖(boxplot)，自動存圖-----')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='Waist_Circum', data=df)
plt.title('Boxplot of Waist Grouped by Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Waist Circumference')
plt.savefig(output_path + '12_boxplot_Waist_Circum_by_diabetes.png')
# plt.show()