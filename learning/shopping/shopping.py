import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    MONTHS = ["jan", "feb", "mar", "apr", "may", "june", "jul", "aug", "sep", "oct", "nov", "dec"]
    VISITOR_TYPE = ["new_visitor", "returning_visitor", "other"]
    BOOLEAN = ["false", "true"]

    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            re = [None] * 17
            re[0] = int(row["Administrative"])
            re[1] = float(row["Administrative_Duration"])
            re[2] = int(row["Informational"])
            re[3] = float(row["Informational_Duration"])
            re[4] = int(row["ProductRelated"])
            re[5] = float(row["ProductRelated_Duration"])
            re[6] = float(row["BounceRates"])
            re[7] = float(row["ExitRates"])
            re[8] = float(row["PageValues"])
            re[9] = float(row["SpecialDay"])
            re[10] = MONTHS.index(row["Month"].lower())
            re[11] = int(row["OperatingSystems"])
            re[12] = int(row["Browser"])
            re[13] = int(row["Region"])
            re[14] = int(row["TrafficType"])
            re[15] = VISITOR_TYPE.index(row["VisitorType"].lower())
            re[16] = BOOLEAN.index(row["Weekend"].lower())
            
            evidence.append(re)
            labels.append(BOOLEAN.index(row["Revenue"].lower()))
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    postotal, negtotal, poscorrect, negcorrect = (0, 0, 0, 0)

    for label, prediction in zip(labels, predictions):
        if label == 1:
            postotal += 1
            if prediction == 1:
                poscorrect += 1
        elif label == 0:
            negtotal += 1
            if prediction == 0:
                negcorrect += 1
        
    sensitivity = poscorrect / postotal
    specificity = negcorrect / negtotal
        
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
