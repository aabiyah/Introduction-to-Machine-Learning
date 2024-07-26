# Terminologies
1. **Clustering** is an unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
2. A **categorical label** has a discrete set of possible values, such as "is a cat" and "is not a cat."
3. A **continuous (regression) label** does not have a discrete set of possible values, which means there are potentially an unlimited number of possibilities.
4. **Discrete** is a term taken from statistics referring to an outcome that takes only a finite number of values (such as days of the week).
5. A **label** refers to data that already contains the solution.
6. Using **unlabeled data** means you don't need to provide the model with any kind of label or solution while the model is being trained.

## Step 1: Define the Problem & Identify the ML Task to Use to Solve it

# Identifying ML Tasks

The presence or absence of labeling in your data often identifies the type of machine learning task.

> Supervised Tasks
Definition: Supervised tasks use labeled data (data containing known solutions or labels).
Example: Predicting the number of snow cones sold based on the average temperature outside.
Details: Data includes temperature and number of snow cones sold. This labeled data is used to train the model for predicting sales, making it a supervised learning task.

> Unsupervised Tasks
Definition: Unsupervised tasks use unlabeled data (data without predefined labels or solutions).
Example: An image of a tree.
Details: The image data is just a matrix of pixels. Since it lacks labels, it is considered unlabeled.

# Clustering in Unsupervised Learning - An Example
Purpose: To identify naturally occurring groupings in unlabeled data.
Example: Identifying book micro-genres.
Scenario: You work for a book recommendation company. Micro-genres like "Teen Vampire Romance" are assumed to exist, but their specific categories are unknown.
Approach: Use clustering to detect groupings in the data based on book descriptions, which can reveal micro-genres.

# Types of Labels in Supervised Learning
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
