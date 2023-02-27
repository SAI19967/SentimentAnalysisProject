from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia=SentimentIntensityAnalyzer()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
  df1=pd.read_excel(r"C:\Users\saite\Downloads\TeamHealthRawDataForDemo.xlsx")
  print(df1.head(10))
  print()
except FileNotFoundError:
    print("read data not correct")

print("sentiment Analyis process started")
df2=df1[['Team','Response']]
print(df2.head(2))
print()
df2['Polarity_score']=df2['Response'].apply(sia.polarity_scores)
print(df2.head(2))
print()
df2['Compound_Score']=df2['Polarity_score'].apply(lambda x:x.get('compound'))
print(df2.head(10))
print()
df3=pd.merge(df1,df2,how='inner')
print(df3.head(10))
print()
print(df3[["Compound_Score"]].describe())


x_axis = ['Team']
y_axis = [' Compound_Score']

plt.bar(df3.Team, df3['Compound_Score'])

plt.title('sentiment score by team')
plt.xlabel('Team')
plt.ylabel('Compound_Score')
print(plt.show())
print()

x_axis = ['Period']
y_axis = [' Compound_Score']

plt.bar(df3.Period, df3['Compound_Score'])

plt.title('sentiment score by period')
plt.xlabel('Period')
plt.ylabel('Compound_Score')
print(plt.show())

sns.set_style("whitegrid")
sns.boxplot(x='Team', y='Compound_Score', notch = True,
            data=df3, showfliers=False).set(title='Sentiment Score by Team')
plt.bar(df3.Period, df3['Compound_Score'])

plt.title('sentiment score by team')
plt.xlabel('Team')
plt.ylabel('Compound_Score')
print(plt.show())


data=df3.Compound_Score.describe()
print(data.loc[['mean','max']])
df3.to_csv(r"C:\\Users\\saite\\Downloads\\nltkoutput.csv",sep=',',header=True,mode="a",index=False)


