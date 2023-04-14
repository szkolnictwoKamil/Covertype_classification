import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# uploading the file
data = pd.read_csv('covtype.data', sep=',', header=None)

# checking for correlation
f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.select_dtypes(['int']).shape[1]), data.select_dtypes(['int']).columns, fontsize=9, rotation=45)
plt.yticks(range(data.select_dtypes(['int']).shape[1]), data.select_dtypes(['int']).columns, fontsize=9)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show() 
#There are a few negatively correlated attributes but they will not be removed

# Let us check for the class distribution
x = Counter(data.iloc[:,-1])
print(x.values())
fig = plt.bar(x.keys(),x.values(), color='r', alpha=0.5, edgecolor='r')
plt.title('The data distribution', fontsize=16)
plt.xlabel('Class', fontweight ='bold', fontsize = 15)
plt.ylabel('Count', fontweight ='bold', fontsize = 15)
plt.show() 
# #The data is not evenly distributed. We should consider using oversampling/undersampling or other methods to overcome class imbalance

#understanding data, searching for outliers, null values and duplicates

print(data.iloc[:,0:10].describe()) #There are visible outliers but for the sake of this task they will not be handled. 
#The example here is the third column where mean is equal 269.428217, 75% of the data is for sure less than 384.000000 but max = 1397

print(data.isnull().values.any())# no missing values
print(data[data.duplicated()]) #no duplicates
