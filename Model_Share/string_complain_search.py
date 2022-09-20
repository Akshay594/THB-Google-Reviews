"""
This file performs the string search for complain/diagnosis.
"""

from diagnosis_entity_extractor import generateResults

input_file = "Validation_Dataset_10.xlsx"

print(generateResults(input_file))