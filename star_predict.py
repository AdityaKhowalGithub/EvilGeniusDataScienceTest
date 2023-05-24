import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# import seaborn as sns


data = pd.read_csv('starcraft_player_data.csv')


# EDA
data.describe()
data.info()

#EDA explores the data through a scatter, box and histogram then shows a correlation matrix
# data['Age'].hist()
# plt.show()

# # Scatter plot
# sns.scatterplot(x='APM', y='LeagueIndex', data=data)
# plt.show()

# # Box plot
# sns.boxplot(x='LeagueIndex', y='Age', data=data)
# plt.show()

# corr = data.corr()
# sns.heatmap(corr)
# plt.show()


# data preproccesing


data = data.dropna()
data = data.fillna(data.mean())
data = pd.get_dummies(data)


le = LabelEncoder()
data['LeagueIndex'] = le.fit_transform(data['LeagueIndex'])



x = data.drop('LeagueIndex', axis = 1)
y = data['LeagueIndex']
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)



#print(y_pred)
results = pd.DataFrame(x_test)
results['TrueRank'] = y_test
results['PredictedRank'] = y_pred
print(results)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')




# Create a DataFrame with the test data and predictions
results = pd.DataFrame(x_test)
results['TrueRank'] = y_test
results['PredictedRank'] = y_pred

# Plot the true and predicted ranks
plt.scatter(results['TrueRank'], results['PredictedRank'])

# Add a diagonal line
plt.plot([0, max(results['TrueRank'])], [0, max(results['TrueRank'])], 'k--')

plt.title('True vs. Predicted Ranks')
plt.xlabel('True Rank')
plt.ylabel('Predicted Rank')
plt.show()
