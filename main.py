import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

comments = pd.read_csv("comentarios.csv")

test, train, sentimental_test, sentimental_train = train_test_split(
    comments.text, comments.label)

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(comments.text)
checked_words = vectorizer.get_feature_names_out()

matrix = pd.DataFrame.sparse.from_spmatrix(bow, columns=checked_words)

test, train, sentimental_test, sentimental_train = train_test_split(
    bow, comments.label)

lr = LogisticRegression()
lr.fit(train, sentimental_train)
accuracy = lr.score(test, sentimental_test)

text = " ".join([comment for comment in comments.text])

words_clouds = WordCloud(width=1024, height=768, collocations=False,
                         background_color="#000").generate(text)

# # Open a plot of the generated image.
# plt.imshow(words_clouds)
# plt.show()

# Save the words in a file
words_clouds.to_file("wordcloud.png")
