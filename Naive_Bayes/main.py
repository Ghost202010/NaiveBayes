import NaiveBayes
import pandas


def main():
    filename = 'Datasets\golf-dataset-categorical.csv'  # Dataset filename
    filename = 'Datasets\Iris.csv'
    train_sample_size = 1  # Training percentage
    # class_column_name = 'Play'
    class_column_name = 'iris'
    dataset = pandas.read_csv(filename, skipinitialspace=True)

    x_train = dataset.sample(frac=train_sample_size)
    y_train = x_train[class_column_name]
    x_train = x_train.drop(columns=class_column_name)

    x_test = dataset.drop(x_train.index)
    y_test = x_test[class_column_name]
    x_test = x_test.drop(columns=[class_column_name])

    verosimilitude_table = NaiveBayes.fit(
        x_train, y_train)
    total_tests = NaiveBayes.tests(x_train, y_train, verosimilitude_table)


if __name__ == '__main__':
    main()
