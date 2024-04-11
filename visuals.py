import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # dataviz
import seaborn as sns
import streamlit as st

df1 = pd.read_csv('C:\\Users\\USER\\Downloads\\immo1.csv')
df = pd.read_csv('C:\\Users\\USER\\Downloads\\immo3.csv')


# Assuming you have a DataFrame named 'data' with numerical columns
numerical_columns = df.select_dtypes(include='number')

fig = plt.figure(figsize=(18, 12))

num_plots = len(numerical_columns.columns)  # Total number of plots
num_rows = (num_plots // 5) + 1  # Calculate the number of rows needed

for i, col in enumerate(numerical_columns.columns):
    plt.subplot(num_rows, 5, i+1)  # Adjusting subplot position
    sns.boxplot(y=numerical_columns[col], showfliers=True)
    plt.title(col)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
st.pyplot(fig)
# Assuming you have a DataFrame nameud 'data' with numerical columns
numerical_columns = df.select_dtypes(include='number')

fig1 = plt.figure(figsize=(18, 12))

num_plots = len(numerical_columns.columns)  # Total number of plots
num_rows = (num_plots // 5) + 1  # Calculate the number of rows needed

for i, col in enumerate(numerical_columns.columns):
    plt.subplot(num_rows, 5, i+1)  # Adjusting subplot position
    sns.boxplot(y=numerical_columns[col], showfliers=True)
    plt.title(col)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
st.pyplot(fig1)


#-----------------------------------------------------------------------------------------------------
# Set seaborn style
sns.set_style("whitegrid")

# 1. Histogram for numeric columns
# fig2 = df.hist(bins=30, figsize=(15, 10))
# plt.title('Histograms of Numeric Variables')
# plt.show()
# st.pyplot(fig2)




# 2. Box plot for comparing livingSpace across typeOfFlat
fig3 = plt.figure(figsize=(10, 6))
sns.boxplot(x='typeOfFlat', y='livingSpace', data=df)
plt.title('Living Space by Type of Flat')
plt.show()
st.pyplot(fig3)

# 3. Scatter plot between livingSpace and baseRent
fig4 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='livingSpace', y='baseRent', data=df)
plt.title('Base Rent vs. Living Space')
plt.show()
st.pyplot(fig4)

# # 4. Pair plot for numeric columns
# fig5 = sns.pairplot(df[['livingSpace', 'baseRent', 'totalRent', 'noRooms', 'serviceCharge']])
# plt.suptitle('Pair Plot of Numeric Variables')
# plt.show()
# st.pyplot(fig5)

# # 5. Heatmap for correlation matrix
# correlation_matrix = df[['livingSpace', 'baseRent', 'totalRent', 'noRooms', 'serviceCharge']].corr()
# fig6 = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Numeric Variables')
# plt.show()
# st.pyplot(fig6)

# 6. Violin plot for totalRent across region
fig7 = plt.figure(figsize=(12, 8))
sns.violinplot(x='regio1', y='totalRent', data=df)
plt.title('Total Rent by Region')
plt.show()
st.pyplot(fig7)

fig8 = plt.figure(figsize=(12, 6))
sns.countplot(x='typeOfFlat', data=df)
plt.title('Count Plot of Type of Flat')
plt.show()
st.pyplot(fig8)

fig9 = plt.figure(figsize=(12, 6))
sns.barplot(x='regio1', y='totalRent', data=df, ci=None, estimator=np.mean)
plt.title('Average Total Rent by Region')
plt.show()
st.pyplot(fig9)

fig10 = plt.figure(figsize=(12, 6))
sns.barplot(x='typeOfFlat', y='livingSpace', data=df, ci=None, estimator=np.mean)
plt.title('Average Living Space by Type of Flat')
plt.show()
st.pyplot(fig10)

# Creating a cross-tabulation between region and hasKitchen
has_kitchen_ct = pd.crosstab(df['regio1'], df['hasKitchen'])

# # Plotting a stacked bar chart
# has_kitchen_ct.plot(kind='bar', stacked=True, figsize=(12, 6))
# plt.title('Proportion of Properties with Kitchen by Region')
# plt.xlabel('regio1')
# plt.ylabel('Count')
# plt.show()
# st.pyplot(has_kitchen_ct)


df['yearConstructed'] = pd.to_datetime(df['yearConstructed'], format='%Y')

# Grouping data by yearConstructed and calculating average totalRent
annual_avg_rent = df.groupby(df['yearConstructed'].dt.year)['totalRent'].mean()

# Plotting a line chart
# plt.figure(figsize=(12, 6))
# plt.plot(annual_avg_rent.index, annual_avg_rent.values, marker='o')
# plt.title('Average Total Rent Over Time')
# plt.xlabel('Year')
# plt.ylabel('Average Total Rent')
# plt.show()
# st.pyplot(annual_avg_rent)


fig11 = sns.jointplot(x='livingSpace', y='totalRent', data=df, kind='reg')
plt.suptitle('Living Space vs. Total Rent')
plt.show()
st.pyplot(fig11)

g = sns.FacetGrid(df, col='regio1', hue='typeOfFlat', col_wrap=3)
g.map_dataframe(sns.scatterplot, x='livingSpace', y='totalRent')
g.add_legend()
plt.figure(figsize=(18,9))
plt.suptitle('Living Space vs. Total Rent Faced by Region and Type of Flat')
plt.show()
st.pyplot(g)


fig12 = plt.figure(figsize=(12, 6))
sns.swarmplot(x='regio1', y='totalRent', data=df)
plt.title('Distribution of Total Rent by Region')
plt.show()
st.pyplot(fig12)
