import pandas
import math
# Training


def fit(x_train, y_train):
    unique_class_values, frequency_table = calculate_frecuency_table(
        x_train, y_train)
    verosimilitude_table = calculate_verosimilitude(
        frequency_table, x_train, y_train)

    return verosimilitude_table


def calculate_frecuency_table(x_train, y_train):
    # Remove spaces
    classes = y_train.astype(str).str.replace(' ', '')
    unique_classes = classes.unique()
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    tablas_de_frecuencia = {}
    for column in all_data.columns:
        if column == y_train.name:
            break
        if all_data[column].dtype in [int, float]:
            tablas_de_frecuencia[column] = all_data[column]
        else:
            all_data[column] = all_data[column].astype(str).str.strip()
            # Frecuency table per column
            # groupby is a Pandas' function, it helps to group data
            # With size(), the functions returns a "Pandas Series" with the number of repeated columns
            # Unstack returns a dataframe, and fill_value helps us to fill missing values
            column_frecuency = all_data.groupby(
                [column, y_train.name]).size().unstack(fill_value=0)
            column_frecuency = column_frecuency.stack()
            for index, a in column_frecuency.items():
                column_frecuency[index] += 1
            column_frecuency = column_frecuency.unstack(fill_value=0)
            # orient='index' is used to return a dictionary by instances (rows)
            tablas_de_frecuencia[column] = column_frecuency.to_dict(
                orient='index')

    return unique_classes, tablas_de_frecuencia


def calculate_verosimilitude(frecuency_table, x_train, y_train):
    # This functions expects a dictionary as a parameter with the following structure:
    # {Outlook:{Sunny:{Yes:2,no:3}, Overcast:{Yes:3,No:1}},Temp:{Hot:{Yes:2,No:2},Mild:{Yes:3,No:2}}}
    verosimilitude = {}
    temp_class_value = {}
    classes = y_train.astype(str).str.replace(' ', '')
    unique_class_values = classes.unique()
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    # first, we get the atributes (columns)
    for attribute in x_train:
        verosimilitude[attribute] = {}
        if all_data[attribute].dtype in [int, float]:
            continuous_verosimilitude()
        else:
            verosimilitude[attribute] = discrete_verosimilitude(
                frecuency_table[attribute], unique_class_values)
    # This is a model output example (It's a dictionary)
    # {Outlook:{Sunny:Yes,Rainy:No,Overcast:Yes}}
    # {Outlook:{Sunny:{Yes:{media:3.2,desviaconEstandar:0.2},no:3}, Overcast:{Yes:3,No:1}},Temp:{Hot:{Yes:2,No:2},Mild:{Yes:3,No:2}}}
    print(verosimilitude)
    return verosimilitude


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


def continuous_verosimilitude():
    pass


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
    print('**** Modelo ****')
    print(pandas.DataFrame(model))
    print('**** Total acertados ****')
    print(succes)
    print('**** Total probado ****')
    print(total_tests)
    print('**** Porcentaje de aciertos ****')
    print((succes*100)/total_tests)
