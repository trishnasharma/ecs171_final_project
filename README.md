# ecs171_final_project

1. Create a GitHub ID

2. Create a GitHub Repository (Public or Private it is up to you. In the end it will have to be Public)

3. Add me to your GitHub Repository (my id is esolares so that I can grade it)

4. Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, size of images, are sizes standardized? do they need to be cropped? normalized? etc.

Our data consists of training and test images separated.

*Character 9 and 25 are not present since 9 - J and Z - 25 are not included since J and Z cannot be represented.*

First, we imported all the necessary libraries for the data exploration step.
<br><br>
We obtained our dataset from kaggle.
<br><br>
We import two different datasets - a train dataset and a test dataset.
<br><br>
!wget - O outputs the url as a file, we do this individually with the train and test dataset.
<br><br>
We obtain the df_train and df_test
<br><br>
df_total is the concatenated version of train and test.
<br><br>
We use plt.figure to plot a histogram to classify our data.
<br><br>
Our data is composed of images. Images that are 28x28 pixels. We have a total of 34627 samples of data. 
<br><br>
We assign y as our label , ie our target.
<br><br>
We then make the heatmap. The heatmap is 24 images where each image has a 784 pixel combination. 
<br><br>
We then plot the example classes before preprocessing. In order to do so, we drop the duplicates and display the images for each of the other labels.
<br><br>
Our data is not standardized, hence we are standardizing it.
We do this using letter = preprocessing.scale(letter)

5. Plot your data. For tabular data, you will need to run scatters, for image data, you will need to plot your example classes.



6. How will you preprocess your data? You should explain this in your Readme.MD file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.



7. Jupyter Notebook data download and environment setup requirements: 



              !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook. Please see HW2 & HW3 for examples.
