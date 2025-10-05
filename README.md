<head> Spam Ham mail Classification</head>
The Hello World of ML projects


This is a Spam Email identifier intially based a custom **Multinomial Naive Bayes'** but i have also added one more method using a Pytorch neural network


<br>
<hr>

**Multinomial Naive Bayes' Model**
    the src directory includes a naive.py file that contains the implementation of said method using only base python functions and numpy. The test data are saved in a .csv format in the data folder, and is extracted using the Pandas Library. Instead of using scikit learn, I have used a custom built Count VEctorizer method in the impelentation Seperate sets were used to train each model. The naive bayes method is a more simpler form but in some cases can outperform the Neural Net Model. In this one, the dataset used to train was AI generated
    <br>
    <hr>
**Pytorch Model**
    This model is a little bit more complex than the Naive Bayes one and hopefully a more accurate version of a spam/ham classifier. For this model, I have used the CountVectorizer method from sci-kit learn instead of using a custom built one. In this one, the dataset used to train has been pulled from kaggle
    



You know the drill, run the code using <it>git pull</it> or like donwlaoad zip and install all required librares from the text and play with it thouself as well :)