# Hand Sign Detection 

Google Colab link: https://colab.research.google.com/github/hirenpateldotdev/ecs171_final_project/blob/main/main.ipynb

### Introduction:
Let’s talk about our project, an American Sign Language recognizer. American Sign Language (ASL) is a language expressed by hand and face movements. Sign language is an essential tool to bridge the communication gap between normal and hearing-impaired people. It is the primary language of many North Americans who are deaf and hard of hearing and is used by many hearing people. Sign Language interpreters interpret ASL in the English language, however, there is a scarcity of interpreters. Therefore, the need for a technology-based system is apparent. It is also a project for social good which is why it was chosen.

### Infographics:

![download](https://user-images.githubusercontent.com/94094315/205823446-8a04257a-189d-4ac6-ae2a-0b652788fa4e.png)

![SL Infographic Population](https://user-images.githubusercontent.com/94094315/205823214-a1240b95-0be2-47f6-9c42-4bf960f33002.jpeg)

Here is a flowchart to show the process of using Machine Learning to identify the hand signs:

![ecs171 drawio](https://user-images.githubusercontent.com/94094315/205795433-c77bdafe-5731-468f-b5fa-22bd19af9518.png)



### Methods and Results:
 #### Data Exploration:
 The data consists of training and test images separated which were combined to train our model efficiently. It was import from kaggle. The data is already formatted to closely match with the classic MNIST and consists 34627 images. Each image or row has label (A = 0 - Z = 25) indicating the letter it represents and a set of 789 pixel values between 0-255 representing 28x28 pixel grayscale image . However, there are no case for letters J (Label 9) and Z (Label 25) because of the gesture motions.
 
 Getting the datasets and exploring data: https://colab.research.google.com/drive/1TV3me4j5mHQLCVA9t7au2VRPObpdV590?authuser=1#scrollTo=SmEfQ8uSxfL6

 Here is one case of each letter which has no gesture motion: 
 
 <img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/letters.png?raw=true" width="400"/>
 
 The American Sign Language MNIST dataset originally included only 1704 images which were cropped to include only hands, resized and then used to create 50+ variations to increse the dataset. Moreover, the dataset is already balance as seen in the bar graph, there are enough cases or images and also same amoutn of cases or images for each letters except for J and X.

 ```
df = pd.concat([df_train,df_test])
display(df.shape)
display(df.describe())


plt.figure(figsize = (10,10)) # Label Count
sns.set_style("darkgrid")
sns.countplot(x=df["label"])
```

The dataset seems balanced as there are enough and almost same cases for each letters without gesture motion.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/Balance%20Dataset.png?raw=true" width="400"/>

The following heatmaps is created by overlapping cases of a letter to check the similarities between the cases of a particular letter.

```
y = df['label']
x = df.values

sizeOfImg = 784
totalLetters = 24
heatmaps = [[0 for pixel in range(sizeOfImg)] for letter in range(totalLetters)]

lableCount = [0]* totalLetters


for k in range(1000):
    row = (df.loc[[k]].values.flatten().tolist())
    label = row[0] - 1
    lableCount[label] += 1
    for i in range(1,sizeOfImg):  
        heatmaps[label][i] = heatmaps[label][i] + row[i]

for k in range(len(lableCount)):
    print(k, " = " ,lableCount[k], " images")
    if (lableCount[k] != 0):
        label = k
        for i in range(sizeOfImg):
            heatmaps[k][i] = heatmaps[k][i] / lableCount[label]

figureHM, axisHM = plt.subplots(nrows=6, ncols=4, figsize=(15,15),)

import cv2

k = 0
for rowsHM in range(6):
    for columnsHM in range(4):
        if (lableCount[k] != 0):
            imgHM = np.reshape(heatmaps[k], (28,28))
            axisHM[rowsHM][columnsHM].imshow(imgHM)
        k+=1
```

<img src="https://user-images.githubusercontent.com/70460449/202991165-d83a4b87-a9c0-4cb0-8a31-b5c646130d23.png" width="400"/>

#### Preprocessing:
The preprocessing only includes scaling the dataset since the data is already modified.
Scaling the dataset includes dividing the pixel values by 255 to make it easy for the model to undeertand the learn.

```
labels = df.label
images = df.iloc[: , 1:]

labels = label_binarizer.fit_transform(labels)

#getting all the letters
letter = df.drop_duplicates(subset = ["label"])
letter = letter.sort_values("label")
letter = letter.drop(columns = ["label"])
letter = letter/255
```

Here is one case of each letter which has no gesture motion after scaling.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/scaled-letter.png?raw=true" width="400"/>

#### Model 1:
In our first model, the neural network flattens the image into a 1-D array for the input.
We trained our first model with 1 hidden layer. The layer sizes were 100(input), 50(hidden), and 25(output). The activation functions used were linear for the hidden layer and ReLu for the input and output. We trained for 10 epochs.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m1_d.png?raw=true" width="400"/>

```
model = Sequential()

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='linear'))
model.add(Dense(25, activation='relu'))

model.add(Dense(classes, activation = 'softmax'))
```
These are the results for Model 1

Fitting the model with our data:

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m1_r.png?raw=true" width="800"/>

Accuracy and Loss Model Complexity graph for training and testing data:

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m1_g.png?raw=true" width="800"/>

The model doesn’t perform well as the accuracy is only 72.6 %

The low accuracy rate was due to the model not being able to identify certain prominent features in the input images. When we flatten the image into a 1-D array, it has no sense of positions and scales invariant structure of the data. Ideally, we want the model to have a 2-D representation of the image where the prominent features are retained, and noise is dropped. Training the model on features would more accurately classify the image as there is less ambiguous information the model is learning from.

#### Model 2:

Convolution Neural Network processes the pixel values from the input images. The input images are processed by multiplying a matrix filter over the image. Once we are done with the convolution layer, we create a 2-D pool layer by applying a function to sections of the convolution.

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m2_d.png?raw=true" width="400"/>

We have a pooling layer of size 3x3 as we have 2 convolutions. The first convolution has 64 layers, and the second one has 36 layers. If we used fewer convolution layers, we wouldn’t be able to identify prominent features causing a lower accuracy rate. m1_g


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

These are the results for Model 2

Fitting the model with our data:

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m2_r.png?raw=true" width="800"/>

Accuracy and Loss Model Complexity graph for training and testing data:

<img src="https://github.com/hirenpateldotdev/ecs171_final_project/blob/main/images/m2_g.png?raw=true" width="800"/>

We were able to get an accuracy of 99.81%, which is highly accurate; thus, we were successful.


### Discussion Sections:
#### Data Exploration
When choosing our dataset, we decided on the Sign Language MNIST datasets from Kaggle since it was a trusted source to get reliable data and it’s important to have good data to build an accurate model. Since there were 24 alphabets in the dataset, we used 24 classes to represent each and had our model classify into those classes. 

#### Preprocessing
To distinguish between the different classes, we used a heat map to analyze the dark and light spots of the images and to use those spots to identify the hand signs from it. The heat map was the best tool to understand the shape of each gesture. Then, we used a label binarizer which changed the label column from values of 1 - 24 to 0 - 1. It would be 1 if the entry matched the letter and 0 otherwise. The label binarizer is a standard for classification problems since it makes it easier for the model to train. 

#### Model 1
When making our model, we initially flattened the image into a 1D array and used that as input for our first model. And we chose the activation function by experimenting with the model and seeing which ones worked the best. However, that didn’t perform as well with an accuracy of 72.6%, and this inaccuracy can be attributed to its inability to identify distinguishing features between the images. From the graph, our model performed well with the training data but not the test, which are signs of underfitting. 

#### Model 2
To improve our model, we made our second model be a CNN with more layers. We trained it with a 2D representation of the images, so that it kept those distinguishing features, leading to more accurate results. We also chose to do a CNN instead since it would process the images without losing its prominent features and give accurate predictions. After evaluating our CNN model, we got an accuracy of 99.81%, and we were finally pleased with our model. In conclusion, we used trial and error to find our final model. With complex data like images, a more complex model is needed, and we didn’t realize this until our first simple model wasn’t as accurate. This made us seek for different machine learning models specialized for image classification and CNN was perfect for that. We also experimented with the convolution layers and their activation functions to perfect our model. 

### Conclusion:
We did a neural network analysis on hand signs from the MNIST kaggle dataset. Before building the actual model we did a bunch of data processing and preprocessing on the datasets. We then built a very basic model with a few layers and modified that to our final model which gave us the best accuracy. We ended up with an accuracy of: 99.8%. Overall we enjoyed brainstorming about different ideas and ML projects to work on and found this one to be very interesting. In the future we could probably try a dataset with different images, for example taking the picture in different lightings and experimenting with that to see if we can train the model better.
