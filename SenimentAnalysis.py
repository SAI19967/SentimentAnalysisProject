from nltk.sentiment.vader import SentimentIntensityAnalyzer
import glob.
sia = SentimentIntensityAnalyzer()

import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_excel(r"C:\Users\saite\Downloads\TeamHealthRawDataForDemo.xlsx")
print(df1.head(10))

print(df1.columns)
df2 = df1[['Team','Response']]
print(df2.head(2))
df2['Polarity_Score']= df2['Response'].apply(sia.polarity_scores)
print(df2.head(5))
print(df2['Polarity_Score'].head(5))
df2['Compound_Score']= df2['Polarity_Score'].apply(lambda x : x.get('compound'))
print(df2['Compound_Score'].head(10))
print(df2.describe())
df3=df2.merge(df1)
dfg = df3.groupby(['Period'])['Compound_Score'].mean()
print(dfg.plot(kind='bar', title='Sentiment Score', ylabel='Mean Compound_Score',
         xlabel='Period', figsize=(6, 5)))

dfg = df3.groupby(['Team'])['Compound_Score'].mean()
print(dfg.plot(kind='bar', title='Compound_Score', ylabel='Mean Compound_Score',
         xlabel='Team', figsize=(6, 5)))


sns.boxplot(x='Team', y='Compound_Score', notch = True,
            data=df3, showfliers=False).set(title='Sentiment Score by Team')

plt.xlabel('Team')
plt.ylabel('Compound_Score')
plt.xticks(rotation=90)