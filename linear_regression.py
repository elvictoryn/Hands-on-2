import numpy as np
import sys

class SimpleLinearRegression:
    def __init__(self):
        # Dataset hardcoded
        self.advertising = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.sales = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self):
        # Cálculo de los coeficientes usando las fórmulas de regresión lineal
        mean_x = np.mean(self.advertising)
        mean_y = np.mean(self.sales)
        
        self.beta_1 = np.sum((self.advertising - mean_x) * (self.sales - mean_y)) / np.sum((self.advertising - mean_x) ** 2)
        self.beta_0 = mean_y - self.beta_1 * mean_x

    def predict(self, X):
        # Predicción usando el modelo ajustado
        return self.beta_0 + self.beta_1 * X

    def print_equation(self):
        # Imprimir la ecuación de regresión
        print(f"Ecuación de Regresión: Sales = {self.beta_0:.2f} + {self.beta_1:.2f} * Advertising")

if __name__ == "__main__":
    # Crear y entrenar el modelo
    model = SimpleLinearRegression()
    model.fit()
    
    # Imprimir la ecuación de regresión
    model.print_equation()
    
    # Leer el valor de Advertising desde la terminal para predecir Sales
    if len(sys.argv) > 1:
        try:
            advertising_value = float(sys.argv[1])
            prediction = model.predict(advertising_value)
            print(f"Predicción para Advertising = {advertising_value} es Sales = {prediction:.2f}")
        except ValueError:
            print("Por favor, ingresa un valor numérico válido para Advertising.")
    else:
        print("Por favor, proporciona un valor de Advertising en la terminal.")
