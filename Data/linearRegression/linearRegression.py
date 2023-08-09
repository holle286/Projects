import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#open csv file but only read the year of the crime column (data is too big to read all of it)
column = ['Year']

#data from https://www.kaggle.com/datasets/nathaniellybrand/chicago-crime-dataset-2001-present
data = pd.read_csv('C:\\Users\\n1ck1\\Desktop\\Projects\\data\\Crimes_-_2001_to_Present.csv', usecols=column)

#list of years from 2001 to 2022
years = [i for i in range(2001, 2023)]


#loop through the data and count the number of crimes per year (2001-2022)
crimes = [0] * 22
for year in data['Year']:
    if year == 2001:
        crimes[0] += 1
    elif year == 2002:
        crimes[1] += 1
    elif year == 2003:
        crimes[2] += 1
    elif year == 2004:
        crimes[3] += 1
    elif year == 2005:
        crimes[4] += 1
    elif year == 2006:
        crimes[5] += 1
    elif year == 2007:
        crimes[6] += 1
    elif year == 2008:
        crimes[7] += 1
    elif year == 2009:
        crimes[8] += 1
    elif year == 2010:
        crimes[9] += 1
    elif year == 2011:
        crimes[10] += 1
    elif year == 2012:
        crimes[11] += 1
    elif year == 2013:
        crimes[12] += 1
    elif year == 2014:
        crimes[13] += 1
    elif year == 2015:
        crimes[14] += 1
    elif year == 2016:
        crimes[15] += 1
    elif year == 2017:
        crimes[16] += 1
    elif year == 2018:
        crimes[17] += 1
    elif year == 2019:
        crimes[18] += 1
    elif year == 2020:
        crimes[19] += 1
    elif year == 2021:
        crimes[20] += 1
    elif year == 2022:
        crimes[21] += 1


#convert lists to numpy arrays for linear regression
npYears = np.array(years)
npCrimes = np.array(crimes)
npYears = npYears.reshape(-1, 1)
model = LinearRegression()

#creates linear regression model
model.fit(npYears, npCrimes)

r_sq = model.score(npYears, npCrimes)

slope = model.coef_
intercept = model.intercept_

#build formula to plot line of best fit
line = slope * npYears + intercept


#make plot with line of best fit
plt.plot(npYears, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope[0], intercept))
#put scatter plot on top of line of best fit
plt.scatter(years, crimes, s = 1.5 , color='blue')
plt.xlabel('Year')
plt.ylabel('# of Crimes')
plt.title('Crimes per Year in Chicago from 2001 to 2022')

#print(slope*2023 + intercept)
# ^^ predicted crimes in 2023 based on model (182466)
plt.show()