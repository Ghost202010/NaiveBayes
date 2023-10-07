import pandas
from scipy.stats import norm
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
        if all_data[attribute].dtype in [int, float]:
            frecuency_table[attribute] = all_data[attribute]
        else:
            all_data[attribute] = all_data[attribute].astype(str).str.strip()
            # Frecuency table per column
            # groupby is a Pandas' function, it helps to group data
            # With size(), the functions returns a "Pandas Series" with the number of repeated columns
            # Unstack returns a dataframe, and fill_value helps us to fill missing values
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


def calculate_verosimilitude(frecuency_table, x_train, y_train):
    # This functions expects a dictionary as a parameter with the following structure:
    # {Outlook:{Sunny:{Yes:2,no:3}, Overcast:{Yes:3,No:1}},Temp:{Hot:{Yes:2,No:2},Mild:{Yes:3,No:2}}}
    verosimilitude = {}
    classes = y_train.astype(str).str.replace(' ', '')
    unique_class_values = classes.unique()
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    # first, we get the atributes (columns)
    for attribute in all_data.columns:
        verosimilitude[attribute] = {}
        if all_data[attribute].dtype in [int, float]:
            verosimilitude[attribute]['avg'] = average_continuous_values(
                x_train, y_train, attribute)
            verosimilitude[attribute]['pstd'] = std_continuous_values(
                x_train, y_train, attribute)
        else:
            all_data[attribute] = all_data[attribute].astype(str).str.strip()
            verosimilitude[attribute] = discrete_verosimilitude(
                frecuency_table[attribute], unique_class_values)
    return verosimilitude

# Not Continuous values (Discrete Values only)


def discrete_verosimilitude(attribute, unique_class_values):
    # Over every atribute we get it's possibles values
    verosimilitude = {}
    temp_class_value = {}
    sum_class = 0
    for one_class in unique_class_values:
        sum_class = 0
        value_attribute_keys = attribute.keys()
        list_keys = list(value_attribute_keys)
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


def tests(x_test, y_test, verosimilitude):
    classes = y_test.astype(str).str.strip()
    unique_class_values = classes.unique()
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_test, classes], axis=1)
    normal_distribution = {}
    success = 0
    total_test = len(all_data)
    for attribute in all_data:
        if all_data[attribute].dtype not in [int, float]:
            all_data[attribute] = all_data[attribute].astype(
                str).str.strip()
    for index, row in all_data.iterrows():
        normal_distribution[index] = {}

        for attribute in all_data.columns:
            for one_class in unique_class_values:
                if all_data[attribute].dtype in [int, float]:
                    if one_class not in normal_distribution[index].keys():
                        normal_distribution[index][one_class] = norm.pdf(row[attribute], loc=verosimilitude[attribute]['avg']
                                                                         [one_class], scale=verosimilitude[attribute]['pstd'][one_class])
                    else:
                        normal_distribution[index][one_class] *= norm.pdf(row[attribute], loc=verosimilitude[attribute]['avg']
                                                                          [one_class], scale=verosimilitude[attribute]['pstd'][one_class])
                else:
                    if one_class not in normal_distribution[index].keys():
                        normal_distribution[index][one_class] = verosimilitude[attribute][row[attribute]][one_class]
                    else:
                        value_attribute_keys = verosimilitude[attribute].keys()
                        list_keys = list(value_attribute_keys)
                        if all(key in unique_class_values for key in list_keys):
                            normal_distribution[index][one_class] *= verosimilitude[attribute][one_class]

                        else:
                            normal_distribution[index][one_class] *= verosimilitude[attribute][row[attribute]][one_class]
        max_normal_distribution = list(normal_distribution[index].values())[0]
        key_max_normal_distribution = list(
            normal_distribution[index].keys())[0]
        for index_keys in normal_distribution[index].keys():
            if normal_distribution[index][index_keys] > max_normal_distribution:
                max_normal_distribution = normal_distribution[index][index_keys]
                key_max_normal_distribution = index_keys
        if all_data.loc[index][classes.name] == key_max_normal_distribution:
            success += 1

        print('Clase esperada')
        print(all_data.loc[index][classes.name])
        print('Clase estimada')
        print(key_max_normal_distribution)
        print('**************************')
    print('aciertos')
    print(success)
    print('Porcentaje de aciertos')
    print((success*100)/total_test)


def print_debug(valor_multiplicado, valor_a_multiplicarse, atributo, clase):
    print('atributo: ' + atributo)
    print('clase: ' + clase)
    print('valor actual')
    print(valor_multiplicado)
    print('valor almacenado')
    print(valor_a_multiplicarse)
