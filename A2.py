import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('future.no_silent_downcasting', True)

def normalizeCol(column, featureRange=(1, 5)):
    column = np.array(column)
    minVal = column.min()
    maxVal = column.max()

    #scale  column values to specified featureRange
    scaled_column = (column - minVal) * (featureRange[1] - featureRange[0]) / (
        maxVal - minVal
    ) + featureRange[0]

    return scaled_column

def standardizeCol(column):
    column = np.array(column)
    mean = np.mean(column)
    std = np.std(column)

    column = (column - mean) / std

    return column

def costFunction(
    X, 
    y, 
    theta, 
    lambda_reg=None
):
    predictions = np.dot(X, theta)
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * len(y))

    if lambda_reg is not None:
        # Add regularization term only if lambda_reg is provided
        cost_reg = lambda_reg * np.sum(np.abs(theta))
        cost += cost_reg
        #print(f"Regularization cost: {cost_reg}")

    #print(f"Cost: {cost}")

    return cost

def multipleDescent(
    X,
    y,
    numFeatures,
    max_iterations=1000,
    learningRate=0.01,
    tolerance=1e-5,
    lambda_reg=None,
    paramPenalty=False,
    verbose=True,
):
    # Convert X to numpy array if it's a pandas Series or DataFrame
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.values

    theta = np.zeros(numFeatures)
    previous_cost = float("-inf")

    lambda_reg = lambda_reg if paramPenalty else None

    for it in range(max_iterations):
        gradient = np.dot(X.T, (np.dot(X, theta) - y)) / len(y)
        if paramPenalty:
            gradient += lambda_reg * np.sign(theta)

        theta -= learningRate * gradient

        current_cost = costFunction(X, y, theta, lambda_reg)

        if np.isnan(current_cost) or np.isinf(current_cost):
            print("Cost function produced NaN or infinity value.")
            break

        if abs(previous_cost - current_cost) < tolerance:
            break

        previous_cost = current_cost

    print(current_cost)  # Print the final cost value after optimization completes

def readCSV(filepath: str, normalize=False, standardize=False):
    dtypes = {
        "price": int,
        "area": int,
        "bedrooms": int,
        "bathrooms": int,
        "stories": int,
        "mainroad": str,
        "guestroom": str,
        "basement": str,
        "hotwaterheating": str,
        "airconditioning": str,
        "parking": int,
        "prefarea": str,
        "furnishingstatus": str,
    }

    df = pd.read_csv(filepath, dtype=dtypes)
    df.columns = [col.strip() for col in df.columns]  # remove extra whitespace

    # Ensure correct inference of data types
    df = df.infer_objects(copy=False)


    # data sanitization - convert yes and no to 1 and 0
    binary_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    df[binary_columns] = df[binary_columns].replace({"yes": 1, "no": 0})
    df.infer_objects(copy=False)
    


    # convert furnishedstatus to decimal values between 0-1
    furnishing_mapping = {"unfurnished": 0, "semi-furnished": 0.5, "furnished": 1}
    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_mapping)
    df = df.infer_objects(copy=False)

    normalize_columns = [
        "stories", 
        "price", 
        "area", 
        "bedrooms", 
        "bathrooms", 
        "parking"]

    for col in normalize_columns:
        if normalize:
            df[col] = normalizeCol(df[col])
        if standardize:
            df[col] = standardizeCol(df[col])

    # assign inputs and output
    X = df[
        [
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "parking",
            "prefarea",
            "furnishingstatus",
        ]
    ]
    y = df["price"]

    # split df into training and validation
    totalRows = len(df)
    trainSize = int(0.8 * totalRows)

    return (X.iloc[:trainSize], y.iloc[:trainSize]), (
        X.iloc[trainSize:],
        y.iloc[trainSize:],
    )

def validateData(Xvalid, yvalid, theta):
    yTrue = []
    for row in Xvalid.values:
        yTrue.append(sum(np.multiply(row, theta)))

    return ((yTrue - yvalid) ** 2).mean()

def plotFeatures(featureNames, theta):
    plt.bar(featureNames, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()
    
def main():
    (xtrain, ytrain), (xvalid, yvalid) = readCSV("./Housing.csv")  # unprocessed
    (nxtrain, nytrain), (nxvalid, nyvalid) = readCSV("./Housing.csv", normalize=True)  # normalized
    (sxtrain, sytrain), (sxvalid, syvalid) = readCSV("./Housing.csv", standardize=True)  # standardized

    # Training scenarios with headers
    print("Raw data training (subset):")
    multipleDescent(
        xtrain[
            ["area", 
            "bedrooms", 
            "bathrooms", 
            "stories", 
            "parking"]], 
            ytrain, 
            5, 
            learningRate=0.01, 
            paramPenalty=False)
    
    print("Raw data training (all):")
    multipleDescent(
        xtrain, 
        ytrain, 
        xtrain.shape[1], 
        learningRate=0.01, 
        paramPenalty=False)

    print("Normalized subset:")
    multipleDescent(
        nxtrain[
            ["area", 
            "bedrooms", 
            "bathrooms", 
            "stories", 
            "parking"]], 
            nytrain, 
            5, 
            learningRate=0.02, 
            paramPenalty=False)

    print("Standardized subset:")
    multipleDescent(
        sxtrain[
            ["area", 
            "bedrooms", 
            "bathrooms", 
            "stories", 
            "parking"]], 
            sytrain, 
            5, 
            learningRate=0.01, 
            paramPenalty=False)

    print("Normalized (all columns):")
    multipleDescent(
        nxtrain, 
        nytrain, 
        nxtrain.shape[1], 
        learningRate=0.01, 
        paramPenalty=False)

    print("Standardized (all columns):")
    multipleDescent(
        sxtrain, 
        sytrain, 
        sxtrain.shape[1], 
        learningRate=0.01, 
        paramPenalty=False)

    print("Training subset of variables (parameter penalty):")
    multipleDescent(
        nxtrain[
        ["area", 
        "bedrooms", 
        "bathrooms", 
        "stories", 
        "parking"]], 
        nytrain, 
        5, 
        learningRate=0.05, 
        paramPenalty=True, 
        lambda_reg=0.01, 
        tolerance=1e-5)

    print("Training all variables (parameter penalty):")
    multipleDescent(
        sxtrain, 
        sytrain, 
        sxtrain.shape[1], 
        learningRate=0.01, 
        paramPenalty=True, 
        lambda_reg=0.2, 
        tolerance=1e-5)

if __name__ == "__main__":
    main()
