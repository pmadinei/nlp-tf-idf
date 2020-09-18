# Introduction
During this project, we try to predict a phone price based on the descriptions and titles in people adverts in Farsi.
## Dataset
The dataset for training and validation is available [HERE](https://github.com/pmadinei/price-tf-idf/blob/master/Docs/mobile_phone_dataset.csv)

# Implementation
## Preprocessing & Previsualizations
first we find that the date which the row has created is weekend or not (friday and thursday)<br>
then we find the time within 24 hours, based on what we expect from the price if one letter has created on night or morning<br> after that, we clean brand to english brands, and then one-hot-encode

## Correlation Heatmap
As it has been clarified in the plot bellow, without description and title columns, the correlations of other columns with price is pretty low. most of correlations are between some of the brands like apple and cities like tehran.

![HeatMap](https://github.com/pmadinei/price-tf-idf/blob/master/Docs/Corrs.png)

## Text Preprocesses
As a result of what has been described, lets preprocess the two desc and title columns. Lets start with normalizing (for better tokenizing) and tokenizing the column strings. by doing that, words will be separated from each other. Now let us normalize the words with the module "informal normalizer" for informal words. After that, the words may concatinate again with each othe. So we tokenize them again. Personally, I guess finding word stems will reduce the precision of our model. as a result of that, I made 2 data frames; one with stemming (dd2) and one without it(dd). Also I removed '\u200c' from both.

## Remove stopwords and Special Chars
Because they are useless to the meaning of sentences and as a result, to prices.

## Just one word columns!
Because only words and the count of them will be important for us, I merged Two "title" and "desc" columns into one column and remove the othe two. Also I filled empty new columns with "missing" for further conciderations.

## Column Normalizations
purpose of normalization : scale numeric data from different columns down to an equivalent scale so that the model doesn’t get skewed due to huge variance in a few columns. for example, prices are vary through dataframe and distances between them are very high. As a result, I splited test columns(unknown prices) and then I used standard scaler normalizer.
After normalization, the RMSE has increased; reason being prices are large numbers and the regression models just try to calculate given numbers to predict the prices. As a result, low numbers could not lead us to the specific large numbers, so that I commented out normalization section.

## Title-Description Feature Extraction : TF-IDF vectorizer
A machine understands only numbers, it does not directly understand letters or text that we as humans can read. That means we need to convert our text and categorical data to numbers. This process is called feature extraction or featurization.
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. In other words, that is intended to reflect how important a word is to a document in a collection or corpus. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. n-gram is a contiguous sequence of n items from a given sample of text or speech. The items here are just words!
I first used sklean's built-in TfidfVectorizer and I have encoded name and item_descriptions into TF-IDF vectors of uni-grams, bi-grams and tri-grams because I guess will be the most important features. I also limited the number of features to 1M in the pursuit of avoiding very high dimensional vectors.
But before doing that, splited words in merged title+description column should be convert to sentences.

## Models & Evaluations
Some the models have been grid searched (with the commented code at the end of this part)...But because I didn't have the time, I just applied for some of them with best params_!
As models below indicate, in most of them, with stemming(train-test2) and without it (train-test1) doesn't make any differences as much, although without stemming the model is Slightly better.
Also we should note that high values of MSE and RMSE is because of great number of prices; for instance, if the model predicts the all prices with 20 Thousand Tomans, It performs pretty well, although MSE will be high. We should rely more on R2 score, which we know what it is from the class!
Fitting times was very high, So I was just able to make limited models.

# Result
Best model created was the Ridge model. Alpha hyperparameter has changed to where test-set had the best scores. Also we should note that Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity; thats why it performs a great model!

* Validation-set:

	Mean Squared Error:  21548831738.122257 
  
	Root Mean Squared Error:  146795.20338935553 
  
	R2 Score:  0.9291079867926274


* Test-set:
	Mean Squared Error:  86184620392.58644 
  
	Root Mean Squared Error:  293572.1723743353 
  
	R2 Score:  0.7124283774278164
  

* Random Predicts on Test-set:

	Mean Squared Error:  775085153598.1973 
  
	Root Mean Squared Error:  880389.2057483424 
  
	R2 Score:  -1.586221233400209
  
Moreover, You can see all reports and code in jupyter notebook in eithor [HERE](https://github.com/pmadinei/price-tf-idf/blob/master/NLP%20for%20Phone%20Price%20Prediction.ipynb) as ipynb or [HERE](https://github.com/pmadinei/price-tf-idf/blob/master/Docs/NLP%20for%20Phone%20Price%20Prediction.html) as HTML in your local computer.
