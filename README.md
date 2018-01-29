# DataScienceWithPython
I will post several mini projects which I have done while studying Data Science course .Please feel free to pull a request

**Session I**

**Data science** is all about data.There are certain steps that are needed to be followed so that we can use the data for 
predictive analysis and other useful works.The order of these steps are very important and by following this order one
can achieve what he dreamt off.

What is Data Science?

Data Science is basically the science behind the data.The data is processed through scientific methods to make some predictions by visualizing it.

Language suitable for Data Science are *Python,R* and other languages.But I will prefer to use *Python* as it has got extensive libraries for data science.


These steps for any data science project which you will work in future-

1. **Aquiring Data**-Collection of data is the first part of any Data Science project.You have to get your dataset in order to start working.There are many sources from which you can get data.One of the sources are-Kaggle,MNIST dataset and the IRIS dataset(preferable for the beginners).

2. **Pre-processing of Data**-This is the second step after data collection where we prepare the data for analysis.The data that we collect often turns out to be inconsistent and so we have to clean the data and this process is called Data Munging.We also have to label the unlabeled data.After this you have to do visualization.

3. **Visualisation of Data**-This is the next step after pre-processing of the data.The correlation between the features will give you a nice insight of the how the features are related to each other.

4. **Analysis of Data**- This is the step where the Machine Learning algorithms come into picture.We will apply machine learning algorithms on the dataset we are having and then predict the accuracy of the model.

5. **Report**- This is the last step where we report the outcome of the model.Like we suggest the list of similar movies upon his past movie experience.

There are two packages in Python which will be very useful in you Data Science Journey.These packages are-Pandas and Numpy.

Numerical Python, or "Numpy" for short, is a foundational package on which many of the most common data science packages are built. Numpy provides us with high performance multi-dimensional arrays which we can use as vectors or matrices.

The key features of numpy are:

    ndarrays: n-dimensional arrays of the same data type which are fast and space-efficient. There are a number of built-in methods for ndarrays which allow for rapid processing of data without using loops (e.g., compute the mean).
    Broadcasting: a useful tool which defines implicit behavior between multi-dimensional arrays of different sizes.
    Vectorization: enables numeric operations on ndarrays.
    Input/Output: simplifies reading and writing of data from/to file.

Check out my numpy notebook for some hands on session.Follow this link-https://github.com/st186/DataScienceWithPython/blob/master/Numpy_Notebook.ipynb

pandas is a Python library for data analysis. It offers a number of data exploration, cleaning and transformation operations that are critical in working with data in Python.

pandas build upon numpy and scipy providing easy-to-use data structures and data manipulation functions with integrated indexing.

The main data structures pandas provides are Series and DataFrames. After a brief introduction to these two data structures and data ingestion, the key features of pandas this notebook covers are:

    Generating descriptive statistics on data
    Data cleaning using built in pandas functions
    Frequent data operations for subsetting, filtering, insertion, deletion and aggregation of data
    Merging multiple datasets using dataframes
    Working with timestamps and time-series data

Check out my pandas notebook for some hands on session.Follow this link-https://github.com/st186/DataScienceWithPython/blob/master/Pandas_Intro.ipynb

Now we have learnt the basics of pandas and numpy and now we will look at Matplotlib library and do some basic exercises.

1. Plotting the data points-bar plot and line plot.
2. Use Mask to filter out some attributes and create nd arrays.
3. Finding correlation between different features.
4. Using Histograms to explore distribution of values.

Please refer to this link-https://github.com/st186/DataScienceWithPython/blob/master/Matplotlib_Intro.ipynb for the Matplotlib tutorial.Please go through it to know the basics of this useful library.

This is the end of Session I where we learnt the basics of numpy and pandas.Hope you enjoyed the session.In the next session we will be working with data preprocessing.See you then, bye.

**Session II**

Welcome to another session,in this session we will talk about something important.In this exercise we will be learning about data pre-processing.Data pre-processing is one of the most important part in building a predictive model.It includes feature selection,working with misssing data,working with categorical data,splitting of data into test and train data.

Now feature selection is basically to nothing but your X(train) which is used to predict Y. In this tutorial you will learn how to select features and then I will teach you what to do with the missing data.We will be using pandas to import the dataset and then with slicing in the main dataset we can create the feature selection nd array.

Now how to deal with the missing data,one way is to remove the row that contain the missing data and then proceed with rest of the data but that is not advicable as we will loose a lot of data in the process.
So what we will do is to use a function called **Imputer** in sckit-learn library which is used to replace the missing data with a strategic value.In this case we have taken the mean of the whole column of which there is a missing data and then we have replaced the missing data with the mean value.Please follow my tutorial for detailed information.

Talking about the categorical data,Machine learning models cannot work on String values like in this case there is one feature called  country which contains values like France,Italy, etc so basically we have to convert them into numeric values.So to convert them into numeric value there we have a function in sckit learn called the **LabelEncoder** which will be able to convert our data into numeric categorical data.

So you can see that the categorical values have been converted to numeric values.

Now one problem that can arise is that the machine learning model can think that as France is 0 and Spain is 2,so it means Spain has bigger or has more priority but that is not the case,so we have to do something so that the model does not misinterprets the data.We will use **OneHotEncoder** to deal with this problem.

Now we will talk about how to split data into test data and train data.Now this is done by using a library in sklearn called cross_validation.Using this we can split the test data and the train data.We will set test data to 20% in this example and rest will be train data.There is a random_state variable which I have set to zero.So don't get into the details of it.Please refer to my jupyter notebook link below for the code.

Now the last topic in the data pre-processing is Feature Scaling.In this section we need to look into some other aspect of Machine Learning.Lets say that the features age and salary are on a different scale and ML algorithms are mostly based on euclidean distances and so the euclidean distance between two salaries will be much larger than that of the age.So basically we have to do somthing to bring them to the same scale and there comes Feature Scaling.We have to normalize the feature values in order to increase the accuracy of your model.So we will use Standard Scalar library in sklearn and using it we will be scaling the X_train and X_test.

**Note- We are not scaling the y_test or y_train because it is categorical but if it was a regression problem then we had to perform the same with y also.**

Another question that may arise is-**Why we use fit_transform() in the X_train and only transform() in X_test during scaling?**
So the answer to that question is that we will first fit and then transform the training data and then using that scaled data we will try to scale the X_test.

Please follow my jupyter notebook for this session's tutorial on this link-https://github.com/st186/DataScienceWithPython/blob/master/Data_Preprocessing_Template.ipynb

This is the end of the second session for now.In the next session we will be working with Regression.Thanks for now.See you then, bye.

**Session III**

Now today we will discuss about **Linear Regression**.Linear Regression aims at fitting the best line to represent the data points in dataset.We will be looking at the dataset which contains feature years of experience into account and calculate the salary of the person.So we will try to fit the best line which will predict the salary correctly.

In Machine Learning a model can accurately predict on train data but may be give wrong assumptions in test data,this is often referred as **Overfitting**.In this example we have used cross validation of sklearn library to divide the dataset into train and test data.We will keep a ratio of 1:3 for test:train.

We will use a visualizing tool matplotlib to visualizing the model fitting.You can learn more through my jupyter notebook on this link-https://github.com/st186/DataScienceWithPython/blob/master/Linear_Regression.ipynb

Now lets visualize the plot from this model-

![alt tag](https://github.com/st186/DataScienceWithPython/blob/master/Images/Firefox_Screenshot_2018-01-29T12-28-36.345Z.png)

So in this plot,the red dots are the datapoints and the blue line fits the train data accurately.

![alt tag](https://github.com/st186/DataScienceWithPython/blob/master/Images/Firefox_Screenshot_2018-01-29T12-22-57.248Z.png)

So in this plot the red dots again are the datapoints and the blue line fits the test data accurately.

So this was Session III,hope you have liked it.See you then bye.
