import NaiveBayes
import pandas


def main():
    filename = 'Datasets\Iris_file.csv'
    train_sample_size = 0.7  # Training percentage
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
    NaiveBayes.tests(x_test, y_test, verosimilitude_table)


if __name__ == '__main__':
    main()
