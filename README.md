# goMLproject
This project explores concurrency in Go using a ML model

# Package & Project Details:
Gonum was chosen for building the ML model for this project. Although GoLearn appeared to be a much more strealined and user friendly package for model building, it wasn't abundatnly clear that goroutines could be utilized with that model building apporach. To ensure that concurrency could be utilized gonum was used and the model was built using matrix algebra to fit a least-squares linear regression.

The project folder contains two programs. The first ML program is housed in the noConcur directory and fulfils the requiement of runing a ML model on the Boston housing data using no concurrency.  Also contained in that directory is a subdirectory called "data". This is a script that loads the dataset from a CSV and is utilized by both programs (See below for comprehensive code details). The other program is housed in the withConcur directory and fulfills the requirment of running the ML program with concurrency. 

# Instructions for Running the Programs:
Using a bash shell, navigate to the directory that contins the executable. 
Within that path run the following:
./noCuncur
    -or-
./withConcur

# Code Walkthrough

## Data Loading

### Loading the Dataset from CSV

The data.go file found in the noConcur directory is used to load the dataset from the boston.csv file. The `LoadDataset` function within the `data` package handles this step, transforming raw data into structured matrices that the machine learning models can use.

#### Function Overview

- **Name**: `LoadDataset`
- **Purpose**: Reads a dataset from a CSV file and transforms it into matrices for the predictors and the target variables.
- **Inputs**:
  - `filePath`: A `string` representing the path to the CSV file containing the dataset.
  - `selectedColumns`: An `[]int` slice indicating the columns to be used as predictors. The target variable is in the last column of the CSV.
- **Outputs**:
  - `*mat.Dense`: A matrix containing the selected predictors from the CSV file.
  - `*mat.Dense`: A matrix containing the target variable from the CSV file.
  - `error`: An error object that will be non-nil if an error occurs during the file reading or data processing steps.

#### Implementation Details

The function begins by attempting to open the specified CSV file, returning an error if the file cannot be accessed. Using the `encoding/csv` package, it reads the entire file into memory and iterates over each row (excluding the header) to extract the necessary data.

For each row, the function:
- Parses the specified columns to use as predictors.
- Extracts the target variable.

This process results in two slices of `float64` values, which are then used to create two `gonum/mat` Dense matrices:
- One matrix for the predictors, with dimensions corresponding to the number of rows (data points) and the number of selected columns (features).
- One matrix for the target variable, with dimensions corresponding to the number of rows and a single column.

---

## Data Preprocessing

### Splitting the Dataset

The Go ML program includes a preprocessing step that prepares the data for the machine learning model. This step involves splitting the input dataset into training and testing sets, a common practice in machine learning to evaluate models accurately. The function responsible for this operation is `splitDataset`.

#### Function Overview

- **Name**: `splitDataset`
- **Purpose**: Splits a dataset into training and testing sets based on a specified ratio.
- **Inputs**:
  - `X`: A `*mat.Dense` matrix representing the feature dataset.
  - `y`: A `*mat.Dense` matrix representing the labels or targets associated with `X`.
  - `ratio`: A `float64` value representing the proportion of the dataset to be used as the training set.
- **Outputs**: Four `*mat.Dense` matrices:
  - `trainX`: Training set features.
  - `trainY`: Training set labels.
  - `testX`: Testing set features.
  - `testY`: Testing set labels.

#### Implementation Details

The function first determines the number of rows to include in the training set based on the provided ratio. It then uses a random permutation of indices to shuffle the dataset, ensuring that the training and testing sets are randomly selected subsets of the original dataset, which helps reduce bias from ordering.

Matrix operations are utilized to create the training and testing subsets for both the features (`X`) and labels (`y`). The function:
- Initializes new matrices for the training and testing sets with appropriate dimensions.
- Copies rows from the original dataset into the new matrices based on the shuffled indices, ensuring that each set receives the correct portion of data as per the specified ratio.

---

## Model Training

### Linear Regression

The Go ML program includes functionality for fitting a linear model to the dataset. This capability is utilized in the `LinearRegression` function, which calculates the regression coefficients that best fit the given data. 

#### Function Overview

- **Name**: `LinearRegression`
- **Purpose**: Fits a linear model to the provided dataset by calculating the regression coefficients that minimize the sum of the squared residuals.
- **Inputs**:
  - `X`: A `*mat.Dense` matrix containing the predictor variables.
  - `y`: A `*mat.Dense` matrix containing the target variable.
- **Output**:
  - `*mat.Dense`: A matrix containing the calculated regression coefficients (theta).

#### Implementation Details

The linear regression implementation follows the normal equation method, which analytically solves for the coefficients that minimize the cost function. The steps involved in this process include:

1. **Computing the Transpose of X and Multiplying by X**: The function first calculates \(X^T X\) (where \(X^T\) is the transpose of \(X\)), resulting in a matrix that is then used to compute the inverse necessary for the normal equation.

2. **Inverting \(X^T X\)**: The function attempts to compute the inverse of \(X^T X\). If the matrix is singular or near-singular, indicating that it cannot be inverted or the inversion is numerically unstable, the function will log a fatal error and terminate.

3. **Multiplying the Transpose of X by y**: It calculates \(X^T y\), preparing the other component required for the normal equation.

4. **Calculating the Regression Coefficients (Theta)**: The function then multiplies the inverse of \(X^T X\) by \(X^T y\) to solve for the regression coefficients, denoted as theta (\(\theta\)).

The output, theta, represents the regression coefficients that best fit the linear model to the data according to the least squares criterion.

---

## Model Evaluation

### Mean Squared Error (MSE) and Akaike Information Criterion (AIC)

Metrics are computed through the `ComputeMSE` and `CalculateAIC` functions, providing insights into the accuracy and efficiency of the fitted models.

#### Computing Mean Squared Error (MSE)

- **Name**: `ComputeMSE`
- **Purpose**: Calculates the Mean Squared Error (MSE) between the actual and predicted values, a common metric for assessing model accuracy.
- **Inputs**:
  - `yActual`: A `*mat.Dense` matrix containing the actual target variable values.
  - `yPredicted`: A `*mat.Dense` matrix containing the predicted values from the model.
- **Output**:
  - `float64`: The calculated MSE value.

**Implementation Details**:

The MSE is calculated by first determining the difference between the actual and predicted values. This difference is then squared element-wise to ensure that all differences contribute positively to the overall error. The MSE is obtained by averaging these squared differences across all observations. This metric provides a straightforward measure of the model's prediction accuracy, with lower values indicating better performance.

#### Calculating Akaike Information Criterion (AIC)

- **Name**: `CalculateAIC`
- **Purpose**: Computes the Akaike Information Criterion (AIC) for the model, a measure of model quality that balances goodness of fit with model complexity.
- **Inputs**:
  - `mse`: The Mean Squared Error of the model, as a `float64`.
  - `n`: An `int` representing the number of observations.
  - `k`: An `int` representing the number of parameters in the model.
- **Output**:
  - `float64`: The calculated AIC value.

**Implementation Details**:

The AIC is calculated using the MSE, the number of observations (`n`), and the number of model parameters (`k`). The formula for AIC highlights the trade-off between the goodness of fit (as indicated by the MSE) and the complexity of the model (reflected by the number of parameters). Lower AIC values suggest a model that better explains the data without unnecessary complexity.

---

## Making Predictions

### Predict Function Overview

The `Predict` function applies the learned regression coefficients (theta) to new predictor variables.

#### Function Overview

- **Name**: `Predict`
- **Purpose**: Generates predictions for a new set of data using the fitted linear model.
- **Inputs**:
  - `X`: A `*mat.Dense` matrix containing the new data's predictor variables.
  - `theta`: A `*mat.Dense` matrix representing the regression coefficients obtained from the model fitting process.
- **Output**:
  - `*mat.Dense`: A matrix containing the predicted values for the target variable based on the input data and the model's coefficients.

#### Implementation Details

The prediction process involves a matrix multiplication operation between the new data matrix `X` and the regression coefficients matrix `theta`. This aligns with the linear regression formula \(Y = X\theta\), where `Y` represents the predicted values.

1. **Preparing the Data**: The function first determines the dimensions of the input data matrix `X` to ensure that the predictions matrix is initialized with the correct number of rows (each corresponding to a prediction for an observation in the input data).

2. **Matrix Multiplication**: The core of the prediction operation, this step multiplies the input data matrix `X` by the coefficients matrix `theta` to produce the predicted values. This operation effectively applies the model to each row of input data, generating a prediction for each observation.

3. **Output**: The result of the matrix multiplication is a new matrix where each element represents the predicted value for the corresponding observation in the input data. This matrix is returned to the caller, allowing for further analysis or evaluation of the predicted values.

---
