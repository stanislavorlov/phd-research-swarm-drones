import pandas as pd
import io

# 1. Підготовка даних (приклад з вашого запиту)
data = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/10 8:26,"2,55",17850
536365,71053,WHITE METAL LANTERN,6,12/1/10 8:26,"3,39",17850
536365,84406B,CREAM CUPID HEARTS COAT HANGER,8,12/1/10 8:26,"2,75",17850
536365,84029G,KNITTED UNION FLAG HOT WATER BOTTLE,6,12/1/10 8:26,"3,39",17850"""

# 2. Завантаження даних 
# Якщо у вас є файл, використовуйте: df = pd.read_csv('vash_fail.csv')
df = pd.read_csv(io.StringIO(data))

# 3. Попередня обробка (заміна коми на крапку для коректного відображення ціни)
df['UnitPrice'] = df['UnitPrice'].str.replace(',', '.').astype(float)

# 4. Відображення основної інформації
print("--- Перші рядки таблиці ---")
print(df.head())

print("\n--- Інформація про типи даних та пропуски ---")
print(df.info())

print("\n--- Вибір підмножини колонок (важливих для експерименту) ---")
subset = df[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']]
print(subset)