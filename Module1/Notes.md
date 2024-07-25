## Definition

Machine learning is part of the broader field of artificial intelligence. This field is concerned with the capability of machines to perform activities using human-like intelligence. Within machine learning there are several different kinds of tasks or techniques:
> 1. In **supervised learning**, every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values. We will explore this in-depth in this lesson.
> 2. In **unsupervised learning**, there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data. We will explore this in-depth in this lesson.
> 3. In **reinforcement learning**, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal. This is a completely different approach than supervised and unsupervised learning. 

## How is Machine Learning Different from Traditional Learning?

In traditional programming, a person manually creates a solution in code, which can be complex and impractical for problems with many edge cases. For example, detecting a cat in an image would require coding for various conditions and scenarios.

In machine learning, the approach is different: a model is trained using data and a training algorithm to handle these complexities. The trained model can then make predictions or identify patterns without needing explicit coding for every possible situation. Essentially, machine learning automates pattern recognition and statistical reasoning to solve problems.

## Nearly all tasks solved with machine learning involve three primary components:
1. **A machine learning model**: Generic program, made specific by data; Eg: a block of code that can be used to solve different kinds of problems.
2. **A model training algorithm**: The procedure to use data to shape the model for some specific use cases refers to training. For this, we need to determine what changes need to me made to the data. This process is called data preprocessin. Then we make changes to the model according to our specific needs, i.e., gently nudging specific parts of the model in a direction that brings the model closer to our goal. We repeat these steps till we think the model is accurate enough. 
3. **A model inference algorithm**: This is the process of using the trained model to solve a task. 

> The 'Clay Analogy' explaining these primary components is as follows - Think about the changes that need to be made. The first thing you would do is inspect the raw clay and think about what changes can be made to make it look more like a teapot. Similarly, a model training algorithm uses the model to process data and then compares the results against some end goal, such as our clay teapot. Make those changes. Now, you mold the clay to make it look more like a teapot. Similarly, a model training algorithm gently nudges specific parts of the model in a direction that brings the model closer to achieving the goal. By iterating these steps over and over, you get closer and closer to what you want, until you determine that you’re close enough and then you can stop.
