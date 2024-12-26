import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\NIT All-Projects\Linear-Poly _Reg\emp_sal.csv")
X = df.iloc[:,1:2].values
Y = df.iloc[:,2].values

#linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

# Linear regresson visualization
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# Poly nominal visulaization
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or Bluff(polynomial Regression")
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()
   
#linear predction
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

#poly predction
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred