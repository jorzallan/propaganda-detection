# import libraries
import os
import pickle as pk
from newspaper import Article

# from nutrition.structure.environment import PROPA_DATADIR, PROPA_MODEL
# from wnserver.response import Label, LabelError, SubFeature
PROPA_DATADIR = './data/'
PROPA_MODEL = ['best_model.sav', 'vectorized.sav']


class PropagandaNews(object):

    def __init__(self):
        self.datadir = PROPA_DATADIR
        self.model_objects = PROPA_MODEL

    def predict_propaganda(self, news_text):
        best_model = pk.load(open(self.datadir + '/' + self.model_objects[0], 'rb'))
        best_vector = pk.load(open(self.datadir + '/' + self.model_objects[1], 'rb'))

        article_tf = best_vector.transform([news_text])
        prop_predict = best_model.predict(article_tf)

        if prop_predict == ['non-propaganda']:
            return Label(0, [
                SubFeature('Left centered', 100, 100,
                           tooltip="Left centered"),

                SubFeature('Right centered', 0, 0,
                           tooltip="Right centered")
            ])
        elif prop_predict == ['propaganda']:
            return Label(100, [
                SubFeature('Left centered', 0, 0,
                           tooltip="Left centered"),

                SubFeature('Right centered', 100, 100,
                           tooltip="Right centered")
            ])
        else:
            return LabelError()


def main():
    url = 'http://www.foxnews.com/politics/2018/06/25/intern-who-cursed-at-trump-is-identified-was-suspended-but-not' \
          '-fired.html '
    news_propaganda = PropagandaNews()

    article = Article(url)
    # print('downloading')

    article.download()
    if article.download_state == 0:  # ArticleDownloadState.NOT_STARTED is 0
        print('Failed to retrieve')

    article.parse()
    label = news_propaganda.predict_propaganda(article.text)
    print(label)


if __name__ == '__main__':
    main()
