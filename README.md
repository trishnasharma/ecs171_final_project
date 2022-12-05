# ecs171_final_project

Google Colab link: https://colab.research.google.com/drive/1TIVu0w7qDO1eshcImmZMPKGkFXgeYDpj?usp=sharing

## Assignment 1 - Data exploration and preprocessing

1. Our data consists of training and test images separated.
Our dataset from kaggle.

<t></t>*<u>Note:</u> Character 9 and 25 are not present since 9 - J and Z - 25 are not included since J and Z cannot be represented.*

2. First, We imported all the necessary libraries for the data exploration step.
3. We import two different datasets - a train dataset and a test dataset.
4. !wget - O outputs the url as a file, we do this individually with the train and test dataset.
5. We obtain the df_train and df_test
6. df_total is the concatenated version of train and test.
7. We use plt.figure to plot a histogram to classify our data.
8. Our data is composed of images. Images that are 28x28 pixels. We have a total of 34627 samples of data. 
9. We then plot the example classes before preprocessing. In order to do so, we drop the duplicates and display the images for each of the other labels.
10. We assign y as our label , ie our target.
11. We then make the heatmap. The heatmap is 24 images where each image has a 784 pixel combination. 
<img src="https://user-images.githubusercontent.com/70460449/202991165-d83a4b87-a9c0-4cb0-8a31-b5c646130d23.png" width="400"/>

12. Our data is not standardized, hence we are standardizing it. We do this using letter = preprocessing.scale(letter)


## Assignment 2 - First Model building and Evaluation

1. We first used a label binarizer. The label column was replaced (which had values between 1 and 24) with a column for each letter which is a 1 if that entry is the letter that the column represents else 0.
2. We first scale our pixel values by dividing each value by 255. ( Pixel values range from 0-255).
3. We trained our first model with 1 hidden layer. The layer sizes were 100(input), 50(hidden), and 25(output).
4. The activation functions used was the linear for the hidden layer and relu for the input and output.
5. We trained for 10 epochs.
6. We use model.predict to create y_pred_test using X_test.
7. We evaluate our model next using classification report.
8. We plot a graph for accuracy and loss visualization using matplotlib.
9. As we see in the graph, we see an underfitting. Our model performs well on the training set but not on the test set.

We seek to improve our model. Our next approach might be a CNN with more layers. The cause for an underfitting maybe that the model is too simple.

## Final Submission

### Introduction:

### Figures:

### Methods:
 #### Data Exploration:
 Our data consists of training and test images separated which were combined to train our model efficiently. It was import from kaggle. The data is already formatted to closely match with the classic MNIST and consists 34627 images. Each image or row has label (A = 0 - Z = 25) indicating the letter it represents and a set of 789 pixel values between 0-255 representing 28x28 pixel grayscale image . However, there are no case for letters J (Label 9) and Z (Label 25) because of the gesture motions. Here is one case of each letter which has no gesture motion: 
 
 <img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/letters.png?raw=true" width="400"/>
 
 The Sign Language MNIST dataset originally included only 1704 images which were cropped to include only hands, resized and then used to create 50+ variations to increse the dataset. Moreover, the dataset is already balance as seen in the bar graph, there are enough cases or images for each letters except for J and X.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/Balance%20Dataset.png?raw=true" width="400"/>

2. First, We imported all the necessary libraries for the data exploration step.
3. We import two different datasets - a train dataset and a test dataset.
4. !wget - O outputs the url as a file, we do this individually with the train and test dataset.
5. We obtain the df_train and df_test
6. df_total is the concatenated version of train and test.
7. We use plt.figure to plot a histogram to classify our data.
8. Our data is composed of images. Images that are 28x28 pixels. We have a total of 34627 samples of data. 
9. We then plot the example classes before preprocessing. In order to do so, we drop the duplicates and display the images for each of the other labels.
10. We assign y as our label , ie our target.
11. We then make the heatmap. The heatmap is 24 images where each image has a 784 pixel combination. 
<img src="https://user-images.githubusercontent.com/70460449/202991165-d83a4b87-a9c0-4cb0-8a31-b5c646130d23.png" width="400"/>

12. Our data is not standardized, hence we are standardizing it. We do this using letter = preprocessing.scale(letter)


#### Preprocessing:

#### Model 1:

#### Model 2:
 
### Results:

### Discussion Sections:

### Conclusion:

### Collabration:


