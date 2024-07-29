## Terminology:
1. **Regression**: A common task in supervised machine learning used to understand the relationship between multiple variables from a dataset.
2. **Continuous**: Floating-point values with an infinite range of possible values. This is the opposite of categorical or discrete values, which take on a limited number of possible values.
3. **Hyperplane**: A mathematical term for a surface that contains more than two planes
4. **Plane**: A mathematical term for a flat surface (like a piece of paper) on which two points can be joined by drawing a straight line.
5. **Bag of words**: A technique used to extract features from text. It counts how many times a word appears in a document (corpus), and then transforms that information into a dataset.
6. **Data vectorization**: A process that converts non-numeric data into a numerical format so that it can be used by a machine learning model.
7. **Silhouette coefficients**: A score from -1 to 1 describing the clusters found during modeling. A score near zero indicates overlapping clusters, and scores less than zero indicate data points assigned to incorrect clusters. A score approaching 1 indicates successful identification of discrete non-overlapping clusters.
8. **Stop words**: A list of words removed by natural language processing tools when building your dataset. There is no single universal list of stop words used by all-natural language processing tools.
9. **Neural Networks**: Composed of many simple models called neurons connected together.
10. **Neurons**: Basic units of a neural network.
11. **Weights**: Trainable parameters that define the connections between neurons.
12. **Convolutional Neural Networks (CNNs)**: A type of neural network specialized for processing and analyzing images.

# Case Study 1: Supervised Learning
## Using machine learning to predict housing prices in a neighborhood, based on lot size and the number of bedrooms.

### Step 1: Define the problem
Can we estimate the price of a house based on lot size or the number of bedrooms?
You access the sale prices for recently sold homes or have them appraised. Since you have this data (rigid labels), this is a supervised learning task. You want to predict a continuous numeric value, so this task is also a regression task.

### Step 2: Build the dataset
For this project, you need data about home prices, so you do the following tasks:
1. Data collection: You collect numerous examples of homes sold in your neighborhood within the past year, and pay a real estate appraiser to appraise the homes whose selling price is not known.
2. Data exploration: You confirm that all of your data is numerical because most machine learning models operate on sequences of numbers. If there is textual data, you need to transform it into numbers. You'll see this in the next example.
3. Data cleaning: Look for things such as missing information or outliers, such as the 10-room mansion. You can use several techniques to handle outliers, but you can also just remove them from your dataset.

You also want to look for trends in your data, so you use data visualization to help you find them. 
You can plot home values against each of your input variables to look for trends in your data. In the following chart, you see that when lot size increases, house value increases.

### Step 3: Model training
Prior to actually training your model, you need to split your data. The standard practice is to put 80% of your dataset into a training dataset and 20% into a test dataset.
Linear model selection
When lot size increases, home values increase too. This relationship is simple enough that a linear model can be used to represent this relationship.
A linear model across a single input variable can be represented as a line. It becomes a plane for two variables, and then a hyperplane for more than two variables. The intuition, as a line with a constant slope, doesn't change.
The Python scikit-learn library (https://scikit-learn.org/stable/) has tools that can handle the implementation of the model training algorithm for you.

### Step 4: Model evaluation
One of the most common evaluation metrics in a regression scenario is called root mean square or RMS. RMS can be thought of roughly as the "average error" across your test dataset, so you want this value to be low.
![RMS Formula]([Link URL](https://www.gstatic.com/education/formulas2/553212783/en/root_mean_square.svg))
In a graph, you want the data points to be as close to the "average" line as possible, which would mean less net error.
You compute the root mean square between your model’s prediction for a data point in your test dataset and the true value from your data. 
In general, as your model improves, you see a better RMS result. You may still not be confident about whether the specific value you’ve computed is good or bad. Many machine learning engineers manually count how many predictions were off by a threshold (for example, $50,000 in this house pricing problem) to help determine and verify the model's accuracy.
To evaluate models, you often use statistical metrics. The metrics you choose are tailored to a specific use case.

### Step 5: Model inference
Deploy the model and readjust as per your needs.

# Case Study 2: Unsupervised Learning
## Using machine learning to isolate micro-genres of books by analyzing the wording on the back cover description.

### Step 1: Define the problem
Is it possible to find clusters of similar books based on the presence of common words in the book descriptions?
You do editorial work for a book recommendation company, and you want to write an article on the largest book trends of the year. You believe that a trend called "micro-genres" exists, and you have confidence that you can use the book description text to identify these micro-genres.

By using an unsupervised machine learning (this machine learning task is especially useful when your data is not labeled) technique called clustering, you can test your hypothesis that the book description text can be used to identify these "hidden" micro-genres.

### Step 2: Build the dataset
Dataset: Gather book descriptions for 800 romance books published this year for your dataset.

Data Exploration, Cleaning, and Preprocessing: 
Vectorization - Convert words into numbers for machine learning.
Data Cleaning - Remove capitalization and standardize verb tense using a Python library. Remove punctuation and stop words (e.g., 'a', 'the').
Data Preprocessing - Perform data vectorization to convert text into a numerical format. Transform text into a bag of words representation for model training.

### Step 3: Model training
Model Training: Use the k-means clustering model to find clusters in your dataset.
Parameter k: Adjust k to specify the number of clusters the model should identify.
Unlabeled Data: Since the data is unlabeled and the number of micro-genres is unknown, train the model with different values of k.
Evaluation: Use metrics to determine the most appropriate value for k. Examples of k values to test include k=2 & k=3.

### Step 4: Model evaluation
Evaluation Metric: Use the silhouette coefficient to assess how well your data is clustered.
Optimal Clusters: Plot the silhouette coefficient to determine the optimal number of clusters. For this case, k=19 is optimal.
Manual Evaluation: Perform manual checks of the model's findings.
Example: One cluster corresponds to "paranormal teen romance," which aligns with industry trends, increasing confidence in the model.
Next Steps: Use the model to explore and find insights for further analysis or articles.

> There are various ways to evaluate the model including Fowlkes-Mallows, V-measure, Silhouette coefficient, Rand index, Completeness, Mutual information, Contingency Matrix, Homogeneity, Pair confusion matrix, Calinski-Harabasz index, Davies-Bouldin index.

### Step 5: Model inference
Cluster Inspection: With k=19, a surprisingly large cluster is found (e.g., fictionalized cluster #7).
Cluster Analysis: Many text snippets in this cluster indicate long-distance relationships.
Next Steps: Identify other self-consistent clusters. Use the insights from these clusters to start writing an article on unexpected modern romance micro-genres.

# Case Study 3: Deep Learning
## Using deep neural networks to analyze raw images from lab video footage from security cameras, trying to detect chemical spills.

### Step 1: Define the problem
Business Scenario: A chemical plant needs fast response for spills and health hazards.
Solution: Use machine learning to automatically detect spills using the plant's surveillance system.
Model Goal: Predict if an image contains a spill or does not contain a spill.

### Step 2: Build the dataset
Collecting Data: Gather images of spills and non-spills in various lighting and environments, including both historical data and staged spills.
Exploring and Cleaning: Ensure spills are clearly visible in the images. Use Python tools to enhance image quality if needed.
Data Vectorization: Convert image data into a numerical format. Each pixel is represented by a number between 0 (black) and 1 (white).
Data Splitting: Divide the image data into a training dataset and a test dataset for model training and evaluation.

### Step 3: Model training
Traditional Approach: Requires hand-engineering features (e.g., edges, corners) and then training a model on these features.
Modern Approach: Deep neural networks, especially Convolutional Neural Networks (CNNs), automate feature learning directly from pixels.
CNNs: A type of neural network designed specifically for processing images. They consist of neurons (simple models) connected by trainable weights.

### Step 4: Model evaluation
Accuracy Limitations: In imbalanced datasets, like detecting spills where the 'no spill' class is prevalent, accuracy may be misleading. A model predicting 'no spill' all the time might still seem accurate but fail to detect actual spills.

Precision and Recall:
Precision: Measures the accuracy of spill predictions. ("Of all predictions of a spill, how many were correct?")
Recall: Measures the model's ability to detect actual spills. ("Of all actual spills, how many were detected?")
Manual Evaluation: To ensure realistic spill detection, compare staged spills with historical records. This helps confirm the model's effectiveness in real scenarios.

> There are various ways to evaluate the model including Accuracy, Precision, Confusion matrix, ROC curve, False Positive Rate, False Negative Rate, Recall, Log Loss, Specificity, F1 Score, Negative Predictive Value.

### Step 5: Model inference
Deployment: The model can be deployed on platforms like AWS Panorama, which supports running machine learning workloads.Class Distribution: Most of the time, the model will predict "does not contain spill."Alert System: When a "contains spill" prediction is made, a paging system can alert the janitorial team to respond promptly.
