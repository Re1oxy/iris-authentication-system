import pandas as pd

class DataLoader:
	"""Loads CSV dataset for iris authentification."""

	def __init__(self, train_path: str, test_path: str, label_column: str):
		self.train_path = train_path
		self.test_path = test_path
		self.label_column = label_column

	def load_train(self):
		df = pd.read_csv(self.train_path)
		X = df.drop(columns= [0])
		y = df[0]
		return X, y

	def load_test(self):
		df = pd.read_csv(self.test_path)
		X = df.drop(columns= [0])
		y = df[0]
		return X, y