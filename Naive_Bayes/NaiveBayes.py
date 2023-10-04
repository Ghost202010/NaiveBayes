import pandas
import math
# Training


def fit(x_train, y_train):
    frequency_table = calculate_frecuency_table(
        x_train, y_train)
    verosimilitude_table = calculate_verosimilitude(
        frequency_table, x_train, y_train)

    return verosimilitude_table


def calculate_frecuency_table(x_train, y_train):
    # Remove spaces
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    frecuency_table = {}
    for attribute in all_data.columns:
        if attribute == y_train.name:
            break
        if all_data[attribute].dtype in [int, float]:
            frecuency_table[attribute] = all_data[attribute]
        else:
            all_data[attribute] = all_data[attribute].astype(str).str.strip()
            # Frecuency table per column
            # groupby is a Pandas' function, it helps to group data
            # With size(), the functions returns a "Pandas Series" with the number of repeated columns
            # Unstack returns a dataframe, and fill_value helps us to fill missing values
            column_frecuency = all_data.groupby(
                [attribute, y_train.name]).size().unstack(fill_value=0)
            column_frecuency = column_frecuency.stack()
            for index, a in column_frecuency.items():
                column_frecuency[index] += 1
            column_frecuency = column_frecuency.unstack(fill_value=0)
            # orient='index' is used to return a dictionary by instances (rows)
            frecuency_table[attribute] = column_frecuency.to_dict(
                orient='index')

    return frecuency_table


def calculate_verosimilitude(frecuency_table, x_train, y_train):
    # This functions expects a dictionary as a parameter with the following structure:
    # {Outlook:{Sunny:{Yes:2,no:3}, Overcast:{Yes:3,No:1}},Temp:{Hot:{Yes:2,No:2},Mild:{Yes:3,No:2}}}
    verosimilitude = {}
    classes = y_train.astype(str).str.replace(' ', '')
    unique_class_values = classes.unique()
    # first, we get the atributes (columns)
    for attribute in x_train.columns:
        verosimilitude[attribute] = {}
        if x_train[attribute].dtype in [int, float]:
            verosimilitude[attribute]['avg'] = average_continuous_values(
                x_train, y_train, attribute)
            verosimilitude[attribute]['pstd'] = std_continuous_values(
                x_train, y_train, attribute)
        else:
            verosimilitude[attribute] = discrete_verosimilitude(
                frecuency_table[attribute], unique_class_values)
    return verosimilitude

# Not Continuous values (Discrete Values only)


def discrete_verosimilitude(attribute, unique_class_values):
    sum_class = 0
    # Over every atribute we get it's possibles values
    verosimilitude = {}
    temp_class_value = {}
    for one_class in unique_class_values:
        sum_class = 0
        if sum_class <= 0:
            sum_class = sum(
                values_attributes[one_class] for values_attributes in attribute.values())
        for value_attribute in attribute:
            if value_attribute not in verosimilitude.keys():
                verosimilitude[value_attribute] = {}
                temp_class_value[value_attribute] = {}
            verosimilitude[value_attribute][one_class] = attribute[value_attribute][one_class] / sum_class
    return verosimilitude

# Numeric average values (Continuous values)


def average_continuous_values(x_train, y_train, attribute):
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    average_column = all_data.groupby(
        y_train.name)[attribute].mean()
    average_column = average_column.to_dict()
    return average_column

# Numeric population standar deviation


def std_continuous_values(x_train, y_train, attribute):
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    std_column = all_data.groupby(
        y_train.name)[attribute].std(ddof=0)
    std_column = std_column.to_dict()
    return std_column


def tests(x_test, y_test, model):
    all_data = x_test
    # Getting the class
    model_class = list(model.keys())[0]
    classes = y_test.astype(str).str.replace(' ', '')
    # Remove spaces with strip
    for attribute in x_test:
        all_data[attribute] = all_data[attribute].astype(str).str.strip()
    all_data = pandas.concat(
        [all_data, classes], axis=1)
    target_class = y_test.name
    success = 0
    for index, row in all_data.iterrows():
        prediction = model[model_class][row[model_class]]
        # To check if we get a succes or a error
        if prediction == row[target_class]:
            success += 1
    # len() return the number of tested instances
    return len(x_test), success


def print_results(succes, model, total_tests):
    pass
