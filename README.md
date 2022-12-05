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
 The data consists of training and test images separated which were combined to train our model efficiently. It was import from kaggle. The data is already formatted to closely match with the classic MNIST and consists 34627 images. Each image or row has label (A = 0 - Z = 25) indicating the letter it represents and a set of 789 pixel values between 0-255 representing 28x28 pixel grayscale image . However, there are no case for letters J (Label 9) and Z (Label 25) because of the gesture motions. Here is one case of each letter which has no gesture motion: 
 
 <img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/letters.png?raw=true" width="400"/>
 
 The American Sign Language MNIST dataset originally included only 1704 images which were cropped to include only hands, resized and then used to create 50+ variations to increse the dataset. Moreover, the dataset is already balance as seen in the bar graph, there are enough cases or images and also same amoutn of cases or images for each letters except for J and X.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/Balance%20Dataset.png?raw=true" width="400"/>

The following heatmaps is created by overlapping cases of a letter to check the similarities between the cases of a particular letter.

<img src="https://user-images.githubusercontent.com/70460449/202991165-d83a4b87-a9c0-4cb0-8a31-b5c646130d23.png" width="400"/>

#### Preprocessing:
The preprocessing only includes scaling the dataset since the data is already modified.

Scaling the dataset includes dividing the pixel values by 255 to make it easy for the model to undeertand the learn.

Here is one case of each letter which has no gesture motion after scaling.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/scaled-letter.png?raw=true" width="400"/>

#### Model 1:
In our first model, the neural network flattens the image into a 1-D array for the input. 
We trained our first model with 1 hidden layer. The layer sizes were 100(input), 50(hidden), and 25(output). The activation functions used were linear for the hidden layer and ReLu for the input and output. We trained for 10 epochs.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m1_d.png?raw=true" width="400"/>

The model doesn’t perform well as the accuracy is only 71 %

```
model = Sequential()

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='linear'))
model.add(Dense(25, activation='relu'))

model.add(Dense(classes, activation = 'softmax'))
```

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m1_r.png?raw=true" width="800"/>

The low accuracy rate was due to the model not being able to identify certain prominent features in the input images. When we flatten the image into a 1-D array, it has no sense of positions and scales invariant structure of the data. Ideally, we want the model to have a 2-D representation of the image where the prominent features are retained, and noise is dropped. Training the model on features would more accurately classify the image as there is less ambiguous information the model is learning from.

#### Model 2:

Convolution Neural Network processes the pixel values from the input images. The input images are processed by multiplying a matrix filter over the image. Once we are done with the convolution layer, we create a 2-D pool layer by applying a function to sections of the convolution. 

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m2_d.png?raw=true" width="400"/>

We have a pooling layer of size 3x3 as we have 2 convolutions. The first convolution has 64 layers, and the second one has 36 layers. If we used fewer convolution layers, we wouldn’t be able to identify prominent features causing a lower accuracy rate. 

In our project, as the images aren’t large, we use a kernel size of  3 x 3 matrix as the filter. Using a filter, we can extract features from an image, such as vertical and horizontal lines, which can be overlooked when we flatten the input image into a 1-D array. This filer matrix is iterated over the entire image, and the result from the matrix multiplication is added to a corresponding matrix.

The Stride is how much the filter is moved after each matrix multiplication. We have the stride set to the default value for the input layer, which is ‘same.’

We also usually need to add padding to the image to ensure that the filter can be performed on the entire image. With our kernel size, we didn’t need to add padding around the image, as the features were all in the center.

```
model2 = Sequential()


model2.add(Conv2D(64, kernel_size=(3,3), activation='relu' , input_shape=(28,28,1)))
model2.add(MaxPool2D(pool_size= (2,2)))

model2.add(Conv2D(36, kernel_size=(3,3), activation='relu' ))

model2.add(Flatten())
model2.add(Dense(100, activation='relu'))
model2.add(Dense(50, activation='linear'))
model2.add(Dense(25, activation='relu'))

model2.add(Dense(classes, activation = 'softmax'))

model2.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',metrics=['accuracy'])
result_2 = model2.fit(X_train, y_train, validation_data= (X_test, y_test), epochs = train_epochs, batch_size = train_batch_size )
```

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m2_r.png?raw=true" width="800"/>

We were able to get an accuracy of 99.81%, which is highly accurate; thus, we were successful. 

 
### Results:

### Discussion Sections:

### Conclusion:
We did a neural network analysis on hand signs from the MNIST kaggle dataset. Before building the actual model we did a bunch of data processing and preprocessing on the datasets. We then built a very basic model with a few layers and modified that to our final model which gave us the best accuracy. We ended up with an accuracy of: (TODO: ENTER ACCURACY HERE). Overall we enjoyed brainstorming about different ideas and ML projects to work on and found this one to be very interesting. In the future we could probably try a dataset with different images, for example taking the picture in different lightings and experimenting with that to see if we can trian the model better.

### Collabration:


