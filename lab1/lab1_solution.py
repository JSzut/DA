import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import statistics as stat

def analysis(data):
    #3
    data.plot(subplots=True, figsize=(8,5))

    #4
    data.hist(bins=30, figsize=(8, 5))

    #5
    data.plot.kde(subplots=True, figsize=(8,5))
    plt.show()

#1
raw = pd.read_csv("Data1.csv")

#2
raw["Unnamed: 0"] = pd.to_datetime(raw["Unnamed: 0"])
raw.set_index("Unnamed: 0", inplace=True)
analysis(raw)

#6
raw2018 = raw.loc[raw.index.year == 2018, ['theta_1','theta_2','theta_3','theta_4']]
analysis(raw2018)

##############

#3
N = 5 + 4
F = [0]*5
L = [1]*4

y = F + L

dic = { 'N' : N, 'y' : y}

#4
model = CmdStanModel(stan_file="bern_1.stan")

#5
sample = model.sample(dic)

#6
theta = sample.stan_variable('theta')
plt.hist(theta, bins=30)

#7
plt.axvline(theta.mean(), color='r')
plt.axvline(stat.median(theta), color='g')
plt.axvline(sample.summary()['5%']['theta'], color='b')
plt.axvline(sample.summary()['95%']['theta'], color='y')

plt.show()
