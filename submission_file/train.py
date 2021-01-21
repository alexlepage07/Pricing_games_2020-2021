"""This script is used to train your model. You can modify it if you want."""

import sys

import pandas as pd

import model

if __name__ == "__main__":
	# This script expects the dataset as a sys.args argument.
	input_dataset = 'training.csv'  # The default value.
	if len(sys.argv) >= 2:
		input_dataset = sys.argv[1]

	# Load the dataset.
	input_data = pd.read_csv(input_dataset)
	Xraw = input_data.drop(columns=['claim_amount'])
	yraw = input_data['claim_amount'].values

	new_model, preprocessing = model.fit_model(Xraw, yraw)

	model.save_model(new_model, preprocessing)
