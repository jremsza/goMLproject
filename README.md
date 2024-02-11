# goMLproject
This project explores concurrency in Go using a machine learning (ML) model.

# Package & Project Details:
Gonum was the library chosen for building the ML model for this project. Although GoLearn appeared to be a much more streamlined and user-friendly package for model building, it wasn't abundantly clear that goroutines could be utilized with that model building approach. To ensure that concurrency could be utilized, Gonum was used, and the model was built using matrix algebra to fit a least-squares linear regression.

The project folder contains two programs. The first ML program is housed in the noConcur directory and fulfills the requirement of running a ML model on the Boston housing data with a subset of predictors using no concurrency. Also contained in that directory is a subdirectory called "data". This is a script that loads the dataset from a CSV and is utilized by both programs (See below for comprehensive code details). The other program is housed in the withConcur directory and fulfills the requirement of running the ML program with concurrency.

# Instructions for Running the Programs:
Using a bash shell, navigate to the directory that contains the program executable. 
Within that path run the following:

    ./noCuncur

or

    ./withConcur

The program will run the ML model on the Boston housing dataset and print the MSE and AIC for a subset of predictors.

# Code Walkthrough

### Loading the Dataset from CSV

The data.go file found in the noConcur directory is used to load the dataset from the boston.csv file for both programs. The `LoadDataset` function within the `data` package handles this step, transforming raw data into structured matrices that the machine learning models can use.

#### Implementation

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

#### Implementation

The function first determines the number of rows to include in the training set based on the provided ratio. It then uses a random permutation of indices to shuffle the dataset, ensuring that the training and testing sets are randomly selected subsets of the original dataset, which helps reduce bias from ordering.

Matrix operations are utilized to create the training and testing subsets for both the features (`X`) and labels (`y`). The function:
- Initializes new matrices for the training and testing sets with appropriate dimensions.
- Copies rows from the original dataset into the new matrices based on the shuffled indices, ensuring that each set receives the correct portion of data as per the specified ratio.

---

## Model Training

### Linear Regression

The Go ML program includes functionality for fitting a linear model to the dataset. This capability is utilized in the `LinearRegression` function, which calculates the regression coefficients that best fit the given data. 

#### Implementation

The linear regression implementation solves for the coefficients that minimize the cost function. The steps involved in this process include:

1. **Computing the Transpose of X and Multiplying by X**: The function first calculates \(X^T X\) (where \(X^T\) is the transpose of \(X\)), resulting in a matrix that is then used to compute the inverse necessary for the normal equation.

2. **Inverting \(X^T X\)**: The function attempts to compute the inverse of \(X^T X\). If the matrix is singular or near-singular, indicating that it cannot be inverted or the inversion is numerically unstable, the function will log a fatal error and terminate.

3. **Multiplying the Transpose of X by y**: It calculates \(X^T y\), preparing the other component required for the normal equation.

4. **Calculating the Regression Coefficients (Theta)**: The function then multiplies the inverse of \(X^T X\) by \(X^T y\) to solve for the regression coefficients, denoted as theta (\(\theta\)).

The output, theta, represents the regression coefficients that best fit the linear model to the data according to the least squares criterion.

---

## Model Evaluation

### Mean Squared Error (MSE) and Akaike Information Criterion (AIC)

Metrics are computed through the `ComputeMSE` and `CalculateAIC` functions, providing insights into the accuracy and efficiency of the fitted models.

**Implementation**:

The MSE is calculated by first determining the difference between the actual and predicted values. This difference is then squared element-wise to ensure that all differences contribute positively to the overall error. The MSE is obtained by averaging these squared differences across all observations. 

**Implementation**:

The AIC is calculated using the MSE, the number of observations (`n`), and the number of model parameters (`k`). The formula for AIC highlights the trade-off between the goodness of fit (as indicated by the MSE) and the complexity of the model (reflected by the number of parameters).

---

## Making Predictions

### Predict Function Overview

The `Predict` function applies the learned regression coefficients (theta) to new predictor variables.

#### Implementation

The prediction process involves a matrix multiplication operation between the new data matrix `X` and the regression coefficients matrix `theta`. This aligns with the linear regression formula \(Y = X\theta\), where `Y` represents the predicted values.

1. **Preparing the Data**: The function first determines the dimensions of the input data matrix `X` to ensure that the predictions matrix is initialized with the correct number of rows (each corresponding to a prediction for an observation in the input data).

2. **Matrix Multiplication**: This step multiplies the input data matrix `X` by the coefficients matrix `theta` to produce the predicted values. This operation applies the model to each row of input data, generating a prediction for each observation.

3. **Output**: The result of the matrix multiplication is a new matrix where each element represents the predicted value for the corresponding observation in the input data. This matrix is returned to the caller, allowing for further analysis of the predicted values.

---

# Concurrency Application

Much of the code seen in the withConcur program is code used from noConcur, but with added concurrency features in the prediction function. This function is designed to be run as a goroutine. It performs its computation concurrently with other goroutines, and it uses a sync.WaitGroup and a channel to synchronize with them.

`PredictConcurrently` takes an input matrix `X`, a pointer of Dense matrix X, and a matrix `theta`, along with a channel for sending the predictions back to the caller, and a `sync.WaitGroup` for managing concurrency.

### Parameters

- `X *mat.Dense`: The input data on which predictions are to be made, structured as a dense matrix.
- `theta *mat.Dense`: The model parameters used for making predictions, also structured as a dense matrix.
- `resultsChan chan<- *mat.Dense`: A channel for sending the prediction results back to the main routine.
- `wg *sync.WaitGroup`: A synchronization that waits for goroutines to finish executing.

#### Implementation

1. `defer wg.Done()` is called when the function starts, and ensures the goroutine has finished.
2. `rows, _ := X.Dims()`: gets the number of rows in the matrix X.
3. `predictions := mat.NewDense(rows, 1, nil)`: This line creates a new Dense matrix with the same number of rows as X and one column. This matrix will hold the result of the matrix multiplication.
4. `predictions.Mul(X, theta)`: This line performs the matrix multiplication.
5. `resultsChan <- predictions`: This line sends the predictions matrix to the resultsChan channel. This allows the result of the computation to be used elsewhere in the program.


# Remarks to Managment

The results of the 100 repetitions benchmark trial demonstrated that the concurrency model is slightly faster compared to the ML approach without concurrency. These tests were run from the command line and saved in their respective directories as a text file. It is my recommendation that the concurrency approach be utilized going forward for building ML models. However, the code necessary to perform these tasks is exhaustive and so care must be taken to determine if the time saved running the code outweighs the time taken to write the code.
