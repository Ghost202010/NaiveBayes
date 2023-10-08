import pandas
from scipy.stats import norm
import time

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
        # Coninuous attribute doesn't have a frecuency table
        if all_data[attribute].dtype in [int, float]:
            frecuency_table[attribute] = all_data[attribute]
        else:
            all_data[attribute] = all_data[attribute].astype(str).str.strip()

            if attribute == y_train.name:
                column_frecuency = all_data.groupby(
                    y_train.name).size()
                frecuency_table[attribute] = column_frecuency.to_dict()
            else:
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

# This functions expects a dictionary as a parameter
def calculate_verosimilitude(frecuency_table, x_train, y_train):
    verosimilitude = {}
    classes = y_train.astype(str).str.replace(' ', '')
    unique_class_values = classes.unique()
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    # first, we get the atributes (columns)
    for attribute in all_data.columns:
        verosimilitude[attribute] = {}
        if all_data[attribute].dtype in [int, float]:
            # avg means average
            # pstd means Population Standar Deviation
            verosimilitude[attribute]['avg'] = average_continuous_values(
                x_train, y_train, attribute)
            verosimilitude[attribute]['pstd'] = std_continuous_values(
                x_train, y_train, attribute)
        else:
            all_data[attribute] = all_data[attribute].astype(str).str.strip()
            verosimilitude[attribute] = discrete_verosimilitude(
                frecuency_table[attribute], unique_class_values)
    return verosimilitude



def discrete_verosimilitude(attribute, unique_class_values):
    # Over every atribute we get it's possibles values
    verosimilitude = {}
    temp_class_value = {}
    sum_class = 0
    for one_class in unique_class_values:
        sum_class = 0
        value_attribute_keys = attribute.keys()
        list_keys = list(value_attribute_keys)
        # Checking if it's the class attribute
        if all(key in unique_class_values for key in list_keys):
            sum_class = sum(attribute.values())
        else:
            sum_class = sum(
                values_attributes[one_class] for values_attributes in attribute.values())
        for value_attribute in attribute:
            if value_attribute not in verosimilitude.keys():
                verosimilitude[value_attribute] = {}
                temp_class_value[value_attribute] = {}
            if all(key in unique_class_values for key in list_keys):
                verosimilitude[one_class] = attribute[one_class] / \
                    sum_class
            else:
                verosimilitude[value_attribute][one_class] = attribute[value_attribute][one_class] / \
                    sum_class
    return verosimilitude

# Numeric - average values (Continuous values)
def average_continuous_values(x_train, y_train, attribute):
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    average_column = all_data.groupby(
        y_train.name)[attribute].mean()
    average_column = average_column.to_dict()
    return average_column


# Numeric - population standar deviation
def std_continuous_values(x_train, y_train, attribute):
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    std_column = all_data.groupby(
        y_train.name)[attribute].std(ddof=0)
    std_column = std_column.to_dict()
    return std_column

# 
def tests(x_test, y_test, verosimilitude):
    classes = y_test.astype(str).str.strip()
    unique_class_values = classes.unique()
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_test, classes], axis=1)
    
    total_len_test = len(all_data)
    
    for attribute in all_data:
        if all_data[attribute].dtype not in [int, float]:
            all_data[attribute] = all_data[attribute].astype(
                str).str.strip()
    probabilities = calculate_probability(all_data,verosimilitude,unique_class_values)
    keys_max_prob = largest_value(all_data,probabilities)
    success = probabilities_result(all_data,keys_max_prob,classes.name)    
        
    print_results(success,total_len_test)

def calculate_probability(all_data,verosimilitude,unique_class_values):
    probability = {}
    for index, row in all_data.iterrows():
        probability[index] = {}
        for attribute in all_data.columns:
            for one_class in unique_class_values:
                if all_data[attribute].dtype in [int, float]:
                    # Normal density function is calculated with norm.pdf()
                    if one_class not in probability[index].keys():
                        probability[index][one_class] = norm.pdf(row[attribute], loc=verosimilitude[attribute]['avg']
                                                                         [one_class], scale=verosimilitude[attribute]['pstd'][one_class])
                    else:
                        probability[index][one_class] *= norm.pdf(row[attribute], loc=verosimilitude[attribute]['avg']
                                                                          [one_class], scale=verosimilitude[attribute]['pstd'][one_class])
                else:
                    if one_class not in probability[index].keys():
                        probability[index][one_class] = verosimilitude[attribute][row[attribute]][one_class]
                    else:
                        value_attribute_keys = verosimilitude[attribute].keys()
                        list_keys = list(value_attribute_keys)
                        if all(key in unique_class_values for key in list_keys):
                            probability[index][one_class] *= verosimilitude[attribute][one_class]
                        else:
                            probability[index][one_class] *= verosimilitude[attribute][row[attribute]][one_class]
    return probability

def largest_value(all_data,probabilities):
    key_max_probability = {}
    for index, row in all_data.iterrows():
        max_normal_distribution = list(probabilities[index].values())[0]
        key_max_probability[index] = list(
            probabilities[index].keys())[0]
        for index_keys in probabilities[index].keys():
            if probabilities[index][index_keys] > max_normal_distribution:
                max_normal_distribution = probabilities[index][index_keys]
                key_max_probability[index] = index_keys
    return key_max_probability

def probabilities_result(all_data,key_max_probability, class_name):
    success = 0
    for index, row in all_data.iterrows():    
        if all_data.loc[index][class_name] == key_max_probability[index]:
            success += 1
        print_stimated_class(all_data.loc[index][class_name],key_max_probability[index],index)
    return success

def print_results(success,total_len_test):
    print('aciertos')
    print(success)
    print('Total de instancias: ' + str(total_len_test))
    print('Porcentaje de aciertos')
    print(round((success*100)/total_len_test, 2))

def print_stimated_class(right_class,stimated,row):
    print('Fila')
    print(row)
    print('Clase esperada')
    print(right_class)
    print('Clase estimada')
    print(stimated)
    print('**************************')
