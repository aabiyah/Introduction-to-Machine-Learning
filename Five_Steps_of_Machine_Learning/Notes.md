# Terminologies
1. **Clustering** is an unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
2. A **categorical label** has a discrete set of possible values, such as "is a cat" and "is not a cat."
3. A **continuous (regression) label** does not have a discrete set of possible values, which means there are potentially an unlimited number of possibilities.
4. **Discrete** is a term taken from statistics referring to an outcome that takes only a finite number of values (such as days of the week).
5. A **label** refers to data that already contains the solution.
6. Using **unlabeled data** means you don't need to provide the model with any kind of label or solution while the model is being trained.
7. **Impute** is a common term referring to different statistical tools that can be used to calculate missing values from your dataset.
8. **Outliers** are data points that are significantly different from other date in the same sample.
9. **Hyperparameters** are settings on the model that are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.
10. A **loss function** is used to codify the model’s distance from this goal.
11. **Training dataset**: The data on which the model will be trained. Most of your data will be here.
12. **Test dataset**: The data withheld from the model during training, which is used to test how well your model will generalize to new data.
13. **Model parameters** are settings or configurations the training algorithm can update to change how the model behaves.

# Step 1: Define the Problem & Identify the ML Task to Use to Solve it

## Identifying ML Tasks

The presence or absence of labeling in your data often identifies the type of machine learning task.

> Supervised Tasks
Definition: Supervised tasks use labeled data (data containing known solutions or labels).
Example: Predicting the number of snow cones sold based on the average temperature outside.
Details: Data includes temperature and number of snow cones sold. This labeled data is used to train the model for predicting sales, making it a supervised learning task.

> Unsupervised Tasks
Definition: Unsupervised tasks use unlabeled data (data without predefined labels or solutions).
Example: An image of a tree.
Details: The image data is just a matrix of pixels. Since it lacks labels, it is considered unlabeled.

## Clustering in Unsupervised Learning - An Example
Purpose: To identify naturally occurring groupings in unlabeled data.
Example: Identifying book micro-genres.
Scenario: You work for a book recommendation company. Micro-genres like "Teen Vampire Romance" are assumed to exist, but their specific categories are unknown.
Approach: Use clustering to detect groupings in the data based on book descriptions, which can reveal micro-genres.

## Types of Labels in Supervised Learning
Label Types and Their Impact

1. Categorical Label
Definition: A label with a discrete set of possible values.
Example: Identifying the type of flower based on a picture.
Details: The model is trained with images labeled with specific flower categories.
Associated Task: Classification.
Note: In classification tasks, the goal is to assign data to predefined categories.

3. Continuous (Regression) Label
Definition: A label that does not have a discrete set of values, often numerical.
Example: Predicting the number of snow cones sold.
Details: The label is a numerical value that could be any number.
Associated Task: Regression.
Note: In regression tasks, the goal is to predict a continuous numerical value.

# Step 2: Build the Dataset

## Data Collection
Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building custom web scraper applications to collect data for your project. You might even have to run a model over your data to generate needed labels. Here is the fundamental question:
Does the data you've collected match the machine learning task and problem you have defined?

## Data Inspection
The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:
1. Outliers
2. Missing or incomplete values
3. Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

## Summary Statistics
Models can make assumptions about how your data is structured.
Now that you have some data in hand, it is a good best practice to check that your data is in line with the underlying assumptions of the machine learning model that you chose.
Using statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insights into the scope, scale, and shape of a dataset.

## Data Visualisation
You can use data visualization to see outliers and trends in your data and to help stakeholders understand your data.

# Step 3: Train the Model

## Splitting Your Dataset

The first step in model training is to randomly split the dataset. This helps to keep some data hidden during training, allowing it to be used for evaluating the model before production. This process helps to test against the bias-variance trade-off.

**DATASET SPLITS**

- **Training Dataset**: The data on which the model will be trained. Typically, this constitutes about 80% of the data.
- **Test Dataset**: The data withheld from the model during training, used to test the model's ability to generalize to new data.

## Putting It All Together and Key Modeling Training Terms

The model training algorithm iteratively updates a model's parameters to minimize a loss function. Here are the key terms:

- **Model Parameters**: Settings or configurations that the training algorithm can update. This can include weights and biases, especially in neural networks.
- **Loss Function**: A function used to measure the model’s distance from the goal. For example, predicting snow cone sales based on weather and defining the loss function as the average distance between predicted and actual sales.

**TRAINING PROCESS**

1. Feed the training data into the model.
2. Compute the loss function on the results.
3. Update the model parameters to reduce the loss.
4. Repeat until a predefined stop condition is met (based on training time, number of cycles, or other mechanisms).

## Things to Remember

- **Machine Learning Frameworks**: Use existing frameworks with built-in implementations of models and training algorithms. Implementing from scratch is generally unnecessary unless developing new models.
- **Model Selection**: Use model selection to determine the best model(s) for the problem. Testing different models is common practice.
- **Hyperparameters**: Settings that are not changed during training but affect the training process, such as the number of clusters.
- **Iteration**: Be prepared to iterate. Pragmatic problem solving in machine learning often involves testing assumptions, trying new approaches, and comparing results.

# Advanced Topics

## Linear Models

Linear models are foundational in machine learning, often covered in introductory coursework. They describe the relationship between input and output numbers through a linear function (e.g., \( y = mx + b \)). In classification tasks, a related model called the logistic model is used, which maps the output of the linear function to the range [0, 1] and is interpreted as the probability of belonging to the target class. Linear models are quick to train and provide a useful baseline for comparing more complex models. For new problems, starting with a simple linear model is often a good approach.

## Tree-Based Models

Tree-based models are the second most common type of model introduced in coursework. They categorize or regress by constructing a large structure of nested if/else conditions. Each if/else block splits the data into different regions, and the training process determines where these splits occur and what values are assigned at each leaf region.

- **Example**: For a light sensor determining sunlight or shadow, a tree might look like: `if (sensor_value > 0.698) then return 1; else return 0;`
- **Popular Tool**: XGBoost is a commonly used implementation of tree-based models, offering enhancements over basic decision trees. It's a good starting point for establishing a baseline.

## Deep Learning Models

Deep learning models are based on the conceptual model of the human brain and consist of neurons connected by weights. Training involves finding optimal values for these weights. There are several notable neural network structures:

- **Feed Forward Neural Network (FFNN)**: The most basic neural network, structured in layers where each neuron in a layer is connected to all neurons in the previous layer.
- **Convolutional Neural Networks (CNN)**: Used primarily for processing grid-like data such as images. They apply nested filters to detect patterns in data.
- **Recurrent Neural Networks (RNN) / Long Short-Term Memory (LSTM)**: Designed for processing sequences of data, such as time series or text, by maintaining state over iterations.
- **Transformer**: A modern architecture that replaces RNN/LSTMs, suitable for large datasets involving sequences of data, often used in natural language processing.

## Machine Learning Using Python Libraries

- **Classical Models**: For linear and tree-based models, scikit-learn is a comprehensive library. Its web documentation is well-organized and a great resource for learning and implementing classical ML techniques.
- **Deep Learning**: For deep learning tasks, the three most common libraries are:
  - **MXNet**
  - **TensorFlow**
  - **PyTorch**

Each of these libraries is feature-rich and suitable for a wide range of machine learning needs.



> Extra: https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html#sphx-glr-auto-examples-applications-plot-outlier-detection-wine-py shows outlier detection in a real dataset.
