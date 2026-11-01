import ssl
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

ssl._create_default_https_context = ssl._create_unverified_context

housing = fetch_california_housing(as_frame=True)
df = housing.frame

x=df.drop(columns=['MedHouseVal'])
y=df['MedHouseVal']

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)

print(f"Zbiór treningowy (Train): {x_train.shape[0]} wierszy")
print(f"Zbiór walidacyjny (Validation): {x_val.shape[0]} wierszy")
print(f"Zbiór testowy (Test): {x_test.shape[0]} wierszy")