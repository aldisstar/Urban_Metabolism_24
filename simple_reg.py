#%%
# Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score
import sympy as sym
#%%

# Open SEM_Variables.xlsx
sem_file_location = "C:/Users/Aldis/Documents/Master Data Science/GitHub/Urban_Metabolism_24/Data/Week_2_Data/SEM_data/SEM_Variables.xlsx"
sem_data = pd.read_excel(sem_file_location)
sem_data = pd.DataFrame(sem_data)
sem_data.head()
# %%

# Add DMI column
sem_data['DMI'] = sem_data['DE_IN (ton/cap)'] + sem_data['IMP_IN (ton/cap)']
sem_data.head()
#%%

# Select same columns to keep
sem_data = sem_data[["Year", "Country", "GDP in 2015 EUR/inhab", "DMI"]]
sem_data.head()
#%%

# The variables
year = sem_data.iloc[:,0]
country = sem_data.iloc[:,1]
GDP = sem_data.iloc[:,2]
DMI = sem_data.iloc[:,3]

# Set a country
Indices_Lithuania = [i for i, country_name in enumerate(country) if country_name=="Lithuania"]
DMI_Lithuania = DMI[Indices_Lithuania]
GDP_Lithuania = GDP[Indices_Lithuania]
year_Lithuania = year[Indices_Lithuania]

# Plot GDP vs Years
plt.plot(year_Lithuania,GDP_Lithuania,'o')
plt.xlabel("Year")
plt.ylabel("GDP [EUR/cap]")
plt.grid()
plt.xlim([1990, 2020]);
plt.ylim([4, 14]);
#%%

# Fitted curve
plt.plot(GDP_Lithuania, DMI_Lithuania,'ro')
p = np.polyfit(GDP_Lithuania, DMI_Lithuania,2)
plt.plot(GDP_Lithuania,np.polyval(p, GDP_Lithuania),color='blue')
plt.xlabel("Gross Domestic Product (GDP) [EUR/cap]")
plt.ylabel("Direct Material Input (DMI) [ton/cap]")
plt.grid();
plt.xlim([4, 14]);
plt.ylim([0, 25]);
plt.legend(['data','Fitted curve']);
#%%

# DataFrame for statsmodels
data_lithuania = pd.DataFrame({'year': year_Lithuania, 'GDP': GDP_Lithuania, 'DMI': DMI_Lithuania})

# Regression
formula = 'DMI ~ GDP'
anova_model = ols(formula, data_lithuania).fit()

# Summary
print(anova_model.summary())

# ANOVA with statsmodels
anova_table = sm.stats.anova_lm(anova_model, typ=2)

# Muestro los resultados de ANOVA
print(anova_table)
#%%

# Parameters
R2 = r2_score(DMI_Lithuania, np.polyval(p, GDP_Lithuania))
print('R2='); print(R2)
print('')
print('[Beta2 Beta1 Beta0]='); print(p)
#%%


# DI vs Years 
x = sym.symbols('x')
y = -23.08 + 6.3 * x - 0.2175 * x**2
DI = sym.diff(y, x)

DI_Lithuania = []

for i in GDP_Lithuania:
  di = DI.subs(x, i)
  DI_Lithuania.append(di)

plt.plot(year_Lithuania, DI_Lithuania,'o')
plt.xlabel("Year")
plt.ylabel("Decoupling Index (DI) [ton/EUR/cap]")
plt.grid();
# %%

# Regression
formula = 'DMI ~ GDP'
anova_model = ols(formula, data_lithuania).fit()

# Summary
print(anova_model.summary())

# ANOVA with statsmodels
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)
# %%
