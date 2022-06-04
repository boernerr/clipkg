import ast
import string
import re
import os
import pandas as pd
import unicodedata
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

from nltk.collocations import QuadgramCollocationFinder
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.cluster import KMeansClusterer
from nltk.corpus import wordnet as wn  # Has now been added in to the nltk_data/corpora directory
from nltk.metrics.association import QuadgramAssocMeasures
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import  LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn import tree
from time import perf_counter

from clipkg import DATA_PATH

# dictionary to determine voice of PRON's in MEMINCIDENTREPORT
PRON_DICT = {'first': ['I', 'i', 'me', 'my', 'mine', 'myself'],
               # 'second': [], # don't think we'll need this
             'third': ['he', 'him', 'his', 'she', 'her', 'hers', 'their', 'their', 'they', 'them'],
             # Want to couple any possessive PRON with any noun in RELATIVE_DICT and search for 'my {noun}'. I'm assuming
             # this should be a good indication of determining if incident is about SOMEONE YOU KNOW
             'possessive': ['my']
}

# dictionary to determine instances of relations mentioned in MEMINCIDENTREPORT
RELATIVE_DICT = {
    'immediate': ['mom', 'mother', 'dad', 'father', 'brother', 'sister', 'son', 'daughter', 'grandmother', 'grandfather',
                  'grandma', 'grandpa', 'parents', 'family', 'husband', 'wife', 'spouse'],
    'distant': ['aunt', 'uncle', 'cousin', 'nephew', 'niece'],
    'friend': ['friend', 'friends', 'boyfriend', 'girlfriend', 'boy friend', 'girl friend']
}

# frequently used words to alias the author of the incident
BUREAU_IDENTIFIERS = ['employee', 'writer']

UNKWN_IDENTIFIERS = ['man', 'woman', 'person', 'boy', 'girl', 'individual', 'individuals'] # The ordering of this list matters. Bc regex isn't applying the pattern to the last item in the list.

# Custom Transformer Classes
class FormatEstimator(BaseEstimator, TransformerMixin):
    '''Class for reading in .csv file and doing VERY BASIC cleanup to include:
    - Dropping last row
    - Dropping duplicate rows
    - Subsetting the data to only include non NULL MEMINCIDENTREPORT rows, because otherwise, the record is all but useless.
    '''

    def __init__(self, text_col,  path_): # =os.path.join(DATA_PATH, 'raw', 'SRSP_SELF_REPORT__lawEnforceInteraction.csv')
        self.file_path = path_
        self.text_col = text_col
        self.df = pd.DataFrame()
        # self._create_df()

    def fit(self, X):
        return self

    def transform(self, X):
        '''Exclude last row, convert DTMSUBMITSR to date dtype.'''
        x_prime = X.copy()

        x_prime = X.iloc[:-1]
        x_prime = x_prime[x_prime[self.text_col].notna()]
        # if 'DTMSUBMITSR' in x_prime.columns:
        #     x_prime.DTMSUBMITSR = pd.to_datetime(x_prime.DTMSUBMITSR)
        x_prime.drop_duplicates(inplace=True)
        return x_prime

    def _create_df(self):
        df = pd.read_csv(self.file_path, engine='python')
        self.df = df

class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    '''Transformer for:
    - Normalizing MEMINCIDENTREPORT column
    - Computing percent of pronouns used in MEMINCIDENTREPORT that are 1st/3rd person.'''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''Create columns for:
        - Count of occurrences for:
            - 1st person pronoun and 3rd person PRON
            - Immmediate and distant relatives
            - friends
            - Unknown persons indicated
        - Ratio of 1st : 3rd person PRON and possessive PRON in conjunction with relatives *not completely accurate bc using 'my'*
        - Flag if text contains reference to the author or other FBI persons
        - Flag if record has a fine/monetary penalty

        We would assume a MEMINCIDENTREPORT with a ratio of more 1st PRON than 3rd PRON would heavily indicate the MEMINCIDENTREPORT
        is a description of the author.
        '''

        x_prime = X.copy()
        x_prime.MEMINCIDENTREPORT = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._normalize_text(x_))

        x_prime['contains_bureau_persons'] = x_prime.MEMINCIDENTREPORT.str.contains('|'.join(i for i in BUREAU_IDENTIFIERS)) # FBI flag; bool
        x_prime['cnt_unknwn_persons'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, UNKWN_IDENTIFIERS, regex=r'[^\w]| ', substr_flag=False))
        x_prime['last_name'] = x_prime.TXTBUREAUNAME.astype(str).apply(lambda x_: self._find_lastname(x_))
        x_prime['contains_author'] = x_prime.apply(lambda x: x.last_name in x.MEMINCIDENTREPORT, axis=1)#.map({True: 1, False: 0}) # last name flag; bool

        # Combine contains_bureau_persons flag with self author flag
        x_prime['contains_author_OR_bureau'] = x_prime.apply(lambda x:  x.contains_bureau_persons > 0 or x.contains_author > 0 , axis=1) # flag combining contains_author | contains_bureau_persons
        x_prime['possessive_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.possessive_entity_checker(x_, regex=r'[^\w]|'))
        x_prime['cnt_third_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, PRON_DICT['third'], regex=r'[^\w]| ', substr_flag=True)) # Notice there is NO space after '|' - this is important!
        x_prime['cnt_first_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, PRON_DICT['first'], regex=r'[^\w]|'))
        x_prime['cnt_immediate_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['immediate'], regex=r'[^\w]| ')) # Notice there IS a space after '|' - this is important!
        x_prime['cnt_dstnt_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['distant'], regex=r'[^\w]| '))
        x_prime['cnt_frnd'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['friend'], regex=r'[^\w]| '))
        x_prime['has_fine'] = x_prime.CURTICKETAMOUNT.apply(lambda x_: 0 if not x_ else 1)

        # Percentages of entities mentioned
        x_prime['pcnt_poss_rela'] = (x_prime.possessive_rela / x_prime.cnt_first_pron).fillna(0) # The percent of possessive relatives ('my ...'). This should highlight records about someone YOU KNOW

        ttl_pron = x_prime.cnt_third_pron + x_prime.cnt_first_pron
        x_prime['pcnt_first_pron'] = round(x_prime.cnt_first_pron/ ttl_pron, 2)
        x_prime.pcnt_first_pron.fillna(0, inplace=True)
        x_prime['pcnt_third_pron'] = round(x_prime.cnt_third_pron / ttl_pron, 2)
        x_prime.pcnt_third_pron.fillna(0, inplace=True)

        # x_prime = x_prime[['TXTINCIDENTTYPE', 'MEMINCIDENTREPORT', 'has_fine', 'contains_author_OR_bureau',
        #                    'contains_bureau_persons', 'contains_author', 'cnt_unknwn_persons', 'pcnt_first_pron', 'cnt_first_pron',
        #                    'cnt_third_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd']]
        x_prime = x_prime[x_prime.TXTINCIDENTTYPE.notna()] # THis doesn't remove the np.nan in TXTINCIDENTTYPE
        return x_prime

    def _normalize_text(self, str_):
        '''Tokenize words and convert to lowercase. As of now, the tokenized column is converted back into a string in
        all lowercase.'''
        tokenized = [word.lower() for word in wordpunct_tokenize(str_)]
        return ' '.join(i for i in tokenized)
        # return [word.lower() for word in wordpunct_tokenize(str_)]

    def entity_checker(self, str_, srch_critria, regex=r'[^\w]| ', substr_flag=False):
        '''Instead of individual functions for each entity, this is one function that will search a string for srch_critria
        given as one of the arguments.

        Param
        -----
        str_ : string
            The string that is being searched. i.e. text field in dataframe.
        srch_critria : list
            list of strings to search in str_. Put another way, returns True if srch_critria in str_
        regex : string
            The regex pattern used in conjuction with srch_critria. This will most likely remain unchanged from default.
        substr_flag : bool
            This flags whether srch_critria can be found as a substring of a larger word within str_ i.e. if True,
            searching for the word 'he' against a string containing the word 'sheriff' WILL return an occurrence because
            'he' is in s(he)riff.
        '''

        if not substr_flag:
            words = [' ?' + i for i in srch_critria]
        # The below regex prevents 'sheriff' from returning True when matching against pronouns. Else, 's(he)riff'
        # would match 'he'
        else:
            words = [' ?[^a-z]' + i for i in srch_critria]
        reg = regex.join(i for i in words)
        return len(re.findall(reg, str_) )

    def possessive_entity_checker(self, str_, srch_critria=None, regex=r'[^\w]|'):
        '''Check for possessive ('my') use of relatives; i.e. 'my uncle'. This is one function that will search a string for srch_critria
        given as one of the arguments.

        Param
        -----
        str_ : string
            The string that is being searched. i.e. text field in dataframe.
        srch_critria : list
            list of strings to search in str_.
        regex : string
            The regex pattern used in conjuction with srch_critria.
        '''

        if srch_critria:
            words = [' ?my' + i for i in srch_critria]
        else:
            words = [' ?my ' + item for sublist in [ i for i in [RELATIVE_DICT[i] for i  in RELATIVE_DICT.keys()]] for item in sublist]

        self.reg = regex.join(i for i in words)
        # print(re.findall(self.reg, str_)) # for testing ONLY
        return len(re.findall(self.reg, str_) )

    def _find_lastname(self, str_, sep=','):
        '''Return just the last name from the TXTBUREAUNAME field. Delimited by a comma (','). This could be initially
        included in the source data by changing the underlying SQL query.

        Params
        ------
        str_ : string
            This indicates the string that sep is searched in.
        sep : string
            This is the string character we are searching for i.e. if sep=',' then we are searching str_ for the position
            of sep.
        '''

        comma_idx = str_.find(sep)
        return str_[: comma_idx].lower()

class TextNormalizer(BaseEstimator,TransformerMixin):
    '''Class for 'normalizing' text by stopword removal and lemmatization.

    Param
    -----
    text_col : str
        The text column of the dataframe to be operated on
    return_type :  str
        Default value='str'. This sets the output of the normalized text column to be either one long string (return_type='str'),
        OR a list of the normalized words.
        This parameter should be adjusted based on the format that downstream steps expect the text data to be in.
        '''

    def __init__(self, text_col,  return_type='str'): #corpus=None,
        self.language = 'english'
        self.stopwords = set(nltk.corpus.stopwords.words(self.language))
        self.lemmatizer = nltk.WordNetLemmatizer()
        # if not text_col:
        #     self.corpus = corpus
        self.text_col = text_col
        self.return_type = return_type

    def is_punct(self,token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self,token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        """
        Normalize the text by lemmatization by removing stopwords and punct.
        This function should be implemented by:
        pd.Series.apply(lambda x: normalize(x))

        Param
        -----
        document : pd.Series
            In this case, we are applying normalize() to a pd.Series in a pd.DataFrame. Each row, then, is a distinct 'document'.
            The pd.Series in question should be one long string.

        """
        doc = pos_tag(wordpunct_tokenize(document))
        # j[0] = token, j[1] = tag
        return [self.lemmatize(j[0].lower(), j[1]) for j in [k for k in doc]
                if not self.is_punct(j[0]) and not self.is_stopword(j[0])]

    def lemmatize(self, token, pos_tag):
        """
        Maps nltk.pos_tag to WordNet tag equivalent.
        Assumes a (token, pos_tag) tuple as input.

        Param
        -----
        token : str
            A token, i.e. a word
        pos_tag : str
            nltk PartOfSpeech tag
        """
        tag = {
             'N': wn.NOUN,
             'V': wn.VERB,
             'R': wn.ADV,
             'J': wn.ADJ
         }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        """Generic fit func() to comply with SKLEARN."""
        return self

    def transform(self, X):
        """Modify the specified text_col by normalizing it. Then convert it from a list to one big string."""

        # This outputs text into a list of tokens
        X[self.text_col] = X[self.text_col].apply(lambda x: self.normalize(x))
        # This outputs text to be one long str
        if self.return_type == 'str':
            X[self.text_col] = X[self.text_col].apply(lambda x: ' '.join(i for i in x))
        return X

class Word2VecNormalizer(TextNormalizer):
    '''Normalize text by lemmatization and stopword removal.

    This class inherits from the TextNormalizer class and add functionality for returning each record within
    self.text_col as a list of sentences (if there are > 0 sentences). The purpose of this class is to preserve the
    sentence structure of self.text_col by indicating the boundary between sentences. The returned list of sentences can
    then be used downstream for tasks such as word co-occurrences, which otherwise would be inaccurate if a simple bag-of-words
    (BOW) methodology was used.
    '''

    def __int__(self):
        super().__init__()

    def sent_tokens(self, X):
        return sent_tokenize(X)

    def normalize(self, document):
        """
        Normalize the text by lemmatization by removing stopwords and punct while maintaining sentence structure.
        This function should be implemented by:
        pd.Series.apply(lambda x: normalize(x))

        Param
        -----
        document : pd.Series
            In this case, we are applying normalize() to a pd.Series in a pd.DataFrame. Each row, then, is a distinct 'document'.
            The pd.Series in question should be one long string.

        """
        # append each sentence to this list so that we can keep distinct sentences
        sents = []
        t0 = perf_counter()
        for sentence in document:
            doc = pos_tag(wordpunct_tokenize(sentence))
            # print([k for k in doc])
            # j[0] = token, j[1] = tag
            sents.append( [self.lemmatize(j[0].lower(), j[1]) for j in [k for k in doc]
                    if not self.is_punct(j[0]) and not self.is_stopword(j[0])])
        return sents
        print(f'{perf_counter() - t0} seconds')

    def transform(self, X):
        X[self.text_col] = X[self.text_col].apply(lambda token: self.sent_tokens(token))
        X[self.text_col] = X[self.text_col].apply(lambda x: self.normalize(x))
        # This outputs text to be one long str
        if self.return_type == 'str':
            X[self.text_col] = X[self.text_col].apply(lambda x: ' '.join(i for i in x))
        return X

class AssumptionLabelTransformer(BaseEstimator, TransformerMixin):

    CODIFY_DICT = {
        1: 'regarding_self',
        2: 'regarding_knwn_prsn',
        3: 'regarding_unkn_prsn'
    }

    def __init__(self, text_col=None):
        self.text_col = text_col
        self.unlbl_idx = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Creates a label for every record that meets the below explicit conditions. Then, returns only records that
         have been assigned a lable. I.e. creating a training set.'''

        # This assumption holds almost ALL the time; Meaning if there is a ticket $ amount, the incident is about the AUTHOR!
        # Only exception found: X.loc[17684]
        # X.loc[X.CURTICKETAMOUNT > 0, 'label'] = 1 # This was old criteria for ABOUT YOURSELF
        X.loc[(X.has_fine == 1) | (X.contains_bureau_persons > 0) | (X.contains_author > 0) | (X.pcnt_first_pron == 0), 'label'] = 1
        X.loc[(X.has_fine == 0) & (X.pcnt_first_pron < .8) & (X.pcnt_poss_rela >= .2) & ((X.cnt_immediate_rela > 0) | (X.cnt_frnd > 0)), 'label'] = 2
        # This is old criteria, but doesn't return as many results as it previously did. THis is because I changed the logic for counting first_pron
        # X.loc[(X.has_fine == 0) & (X.pcnt_first_pron < .30) & (X.pcnt_poss_rela >= .5) & ((X.cnt_immediate_rela > 0 ) | (X.cnt_frnd > 0 )), 'label'] = 2
        X.loc[(X.cnt_unknwn_persons > 4) & (X.contains_author == 0) & (X.pcnt_poss_rela > 0), 'label'] = 3 # Changed X.cnt_unknwn_persons > 2 to > 4
        self.unlbl_idx = X[X.label.isna()].index

        return X[X.label.notna()]

class OneHotVectorizer(BaseEstimator, TransformerMixin):
    '''Class for one-hotting words(i.e. features) into an array in order to feed to a model downstream.'''

    def __init__(self, text_col ): # removing param: text_col
        self.vectorizer = CountVectorizer(input='content', decode_error='ignore', binary=True)
        self.text_col = text_col

    def fit(self, X, y=None):
        '''Generic fit function.'''
        return self

    def transform(self, X):
        '''Vectorize self.text_col into an array of arrays. This is so we can feed the AoA to a model. '''

        freqs = self.vectorizer.fit_transform(X[self.text_col]) # Original line of code
        # freqs = self.vectorizer.fit_transform(X) # This line would be used if we don't have a text_col as a param.
        return [freq.toarray()[0] for freq in freqs]

class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='most_frequent'):
        self._col_names = None
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

    def fit(self, X, y=None):
        self._col_names = X.columns
        self.imputer.fit(X)

    def transform(self, X):
        X_ = self.imputer.transform(X)
        X_ = pd.DataFrame(X_, columns = self._col_names)
        return X_

class DownSamplerTransformer(BaseEstimator, TransformerMixin):
    '''Once we have labeled data, this class will allow us to select the proportion of each label to be included in a
    train set. This is necessary because we want to approximate the distribution of the population in the sample used
    for training.
    '''
    CODIFY_DICT = {
        1: 'regarding_self',
        2: 'regarding_knwn_prsn',
        3: 'regarding_unkn_prsn'
    }
    BASELINE_WEIGHTS = {1: .60, 2: .30, 3: .10}
    EVEN_WEIGHTS = {1: .34, 2: .33, 3: .33}

    def __init__(self, placeholder_weights=None):
        self.placeholder_weights = placeholder_weights
        self.smpl_distr = None # This variable can be called from the pipeline to check sample distribution!
        # if self.placeholder_weights is None:
        #     self.smpl_distr = self.BASELINE_WEIGHTS
        # else:
        #     self.sampl_distr = self.placeholder_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None): #, colmn='label', smpl_distr=None, rand=96
        '''Set the distribution of a training set so that the least frequent labeled class' proportion is set first.
        Note: This will break IF the percentages present in smpl_distr.values() for a given class label evaluates to a
        number higher than the amount of records for that class in X. To fix, this we could simply allow sampling WITH
        replacement.

        Params
        ------
        X : pd.Dataframe
            pretty self explanatory, but should include a column with labels
        colmn : str
            a column in X that houses the labels for each row
        dictLike : dict()
            a dictionary that should have class labels as keys and percentages as values i.e.
            The percentages should add up to 1

        Ideal baseline distribution for smpl_distr:
        {
        1: .50, # labeled as about YOURSELF
        2: .40, # labeled as about SOMEONE YOU KNOW
        3: .10} # labeled as about RANDO
        '''

        if self.placeholder_weights is None:
            self.smpl_distr = self.BASELINE_WEIGHTS
        else:
            self.smpl_distr = self.placeholder_weights
        # else:
        #     smpl_distr = self.EVEN_WEIGHTS
        colmn='label'
        rand=96
        xprime = pd.DataFrame()

        labels_cnts = X[colmn].value_counts().to_dict()
        # The labeled class with the lowest count
        least_value = min(labels_cnts, key=labels_cnts.get)
        # This makes use of ALL of the records of the smallest labeled class by setting the count of this class' records to
        # the proportion of the class in smpl_distr. The shape[0] of xprime is set based on least_value/ the value of the key for
        # least value in smpl_distr
        try:
            df_shape0 = round(labels_cnts[least_value] / self.smpl_distr[least_value], 0)
        except KeyError as e:
            print(f'KeyError of {e}: The dictLike object key(s) do not match the values within colmn. Make sure the value '
                  f'of {e} is present in the keys of dictLike argument!')

        for label in labels_cnts.keys():
            sample_amt = int(round(df_shape0 * self.smpl_distr[label], 0))
            print(f'Class label: {label} "{self.CODIFY_DICT[label]}" results in {sample_amt} rows')
            xprime = xprime.append(X[X[colmn] == label].sample(n=sample_amt, random_state=rand))

        xprime.reset_index(inplace=True, drop=True)
        return xprime

class ColumnSelector(BaseEstimator, TransformerMixin):
    '''Select only the columns to go to the next step. I'm assigning this functionality to an independent class because
    I want to know exactly where in my pipeline I'm performing column selection. Otherwise, this step would be lost in some other
    transform class and may be hard to find if errors occur down stream.'''

    def __init__(self, column_list):
        self.column_list = column_list

    def fit(self, X, y=None):
        '''y should be irrelevant here, we're only subselecting columns in transform(), i.e not really doing and transforms.'''
        return self

    def transform(self, X,  y=None):
        '''We're just subselecting columns, NOT actually doing transforms on the data.'''
        return X[self.column_list]

class TextCountVectorizer(BaseEstimator, TransformerMixin):
    '''Transformer class for vectorizing the data by implementing the CounterVectorizer() class from sklearn.
    '''
    def __init__(self, text_col, max_df=.95, min_df=20):
        self.text_col = text_col
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        '''max_df param removes words that occur > the percentage (or integer) supplied.
        min_df param removes words that occur < the percentage (or integer) supplied.'''
        vectorizer = CountVectorizer(input='content', max_df=self.max_df, min_df=self.min_df)

        xprime = vectorizer.fit_transform(X[self.text_col])
        self.feature_names = vectorizer.get_feature_names()
        return xprime

class StopWordExtractorTransformer(BaseEstimator, TransformerMixin):
    '''Custom transformer class to create 'stopwords' relative to the input corpus. This is completed by first removing
    generic stop words listed in nltk.corpus.stopwords. Then, a Counter() is used to simply count frequencies of all tokens
    in the corpus. Tokens with the highest (n) frequency are returned as 'stopwords'!

    There should be some kind of normalization step prior to instantiating this step. I.e. TextNormalizer(). This removes
    common English stopwords that won't give us any insight into the underlying words. Once we have removed common stops,
    we can get an idea of the language used by returning the words with the highest frequency in a corpus.
    Example:

    sample_pipe = Pipeline([('normalize', TextNormalizer(text_col='__column__')),
                 ('stop_words', StopWordExtractorTransformer(text_col='__column__'))])
    '''

    def __init__(self, text_col):
        self.text_col = text_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Return a Counter() obj with word frequencies.
        '''
        vocab_raw = X[self.text_col]
        # vocab_raw = [word for lst in vocab_raw for word in lst]
        # Hacky way to do this, but basically have to flatten the list of words twice, which is what below accomplishes:
        vocab_raw = [word for sentence in [word for lst in vocab_raw for word in lst] for word in sentence]
        return Counter(vocab_raw).most_common()

class SignificantCollocationsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, text_col,
                 ngram_class=QuadgramCollocationFinder, # Play with n-gram
                 metric=QuadgramAssocMeasures.pmi): #
        self.ngram_class = ngram_class
        self.metric = metric
        self.text_col = text_col

    def fit(self, X, y=None):
        # ngrams = self.ngram_class.from_documents(X)
        # self.scored_ = dict(ngrams.score_ngrams(self.metric))
        return self

    def transform(self, X):

        gram = X[self.text_col].apply(lambda x: self.ngram_class.from_words(x))
        X['n_grams'] = gram.apply(lambda x: dict(x.score_ngrams(self.metric)))
        X['nbest_ngram'] = gram.apply(lambda x: x.nbest(QuadgramAssocMeasures.raw_freq, 10))
        X['top_collocations'] = X['n_grams'].apply(lambda x: self.take_top_n_dict(x))

        # X['n_grams'] = dict(gram.score_ngrams(self.metric))
        # Below from book:
        # for doc in X:
        #     ngrams = self.ngram_class.from_words(X)
        # return {
        #     ngram: self.scored_.get(ngram, 0.0)
        #     for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 50)
        # }
    @staticmethod
    def take_top_n_dict(d):
        '''Only return the ngram keys that have the highest score. Many ngrams will have the same score, so this should
        return several ngrams.'''
        ngram_list = [(k) for k, val in d.items() if val == max(d.values())]

        return ' '.join(i for i in [word for list in ngram_list for word in list])

class FormatEstimatorRussia(FormatEstimator):
    '''Class for reading in .csv file and doing VERY BASIC cleanup to include:
    - Dropping last row
    - Dropping duplicate rows
    - Subsetting the data to only include non NULL MEMINCIDENTREPORT rows'''

    def __init__(self, text_col, path_): #  = os.path.join(DATA_PATH, 'raw', 'earcaira__all_derog_event_text_dets_2_RUSSIA_query.csv')
        super().__init__(text_col, path_)
        self.file_path = path_
        # self.df = pd.DataFrame()
        # self._create_df()

    def transform(self, X):
        '''Exclude last row, convert DTMSUBMITSR to date dtype.'''
        # xprime = X.copy()
        xprime = X.iloc[:-1]
        xprime = xprime[xprime[self.text_col].notna()]
        xprime.EVENT_DATE = pd.to_datetime(xprime.EVENT_DATE)
        xprime.drop_duplicates(inplace=True)
        return xprime
    #
    # def _create_df(self):
    #     df = pd.read_csv(self.file_path, engine='python')
    #     self.df = df

class CondenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols=['EVENT', 'ALLEGATION_ID', 'ALLEGATION', 'DISPOSITION_RMKS']):
        self.cols = cols

    def fit(self, X, y= None):
        return self

    def transform(self, X):
        xprime = X[self.cols].copy()
        xprime = xprime[(xprime.EVENT == 'SRSP INCIDENT') | (xprime.EVENT== 'SIRS')]
        xprime.drop_duplicates(inplace=True)
        return xprime

class WordMatrixTransformer(BaseEstimator, TransformerMixin):
    '''Preprocess data to be ingested into a neural network.

    This class ingests text as a list of sentences, and outputs sparse matrices based upon the co-occurrence of words
    within the specified window parameter. I.e. if window=2, for each given word (the 'center word') we will denote a
    co-occurrence with other words that occur positionally within [word -2 and word +2].

    Example sentence:
        'George veraciously confirmed the fact that Al Gore is undeniably man-bear-pig.'

        We will assume that our center word from the above = 'fact' and the window parameter = 2. After stopword removal,
        the returned list of co-occurrences will be as follows:
        [
            ['fact', 'confirmed'],
            ['fact', 'veraciously'],
            ['that','fact'],
            ['Al','fact']
        ]
    '''
    test_var = 0

    def __init__(self, window=2, text_col='ALLEGATION'):
        self.text_col = text_col
        self.window = window
        # self.word_lists = [] # List of bigrams for a center word and all words within its window; not using this now
        # self.all_sentences = [] # Only used for testing to ensure sentences are correctly created
        self.all_sentences_encoded = []
        self.uniq_words = None
        self.uniq_wrd_dict = dict()
        self.idx2word = dict()
        self.bag_of_words = [] # Don't really need this as an instance variable
        self.n_words = 0 #len(self.uniq_words), count of uniq words
        self.word_counts = None # The frequency for all words within the entire corpus
        WordMatrixTransformer.test_var += 1 # Test if this increments self.test_var by 1 on each instantiation of this class

    # def _master_sentence_list(self, text):
    #     '''Create master list of all sentences in data set/corpus. This is really only for testing purposes to ensure
    #     sentences are constructed properly.'''
    #     for sentence in text:
    #         self.all_sentences.append(sentence)

    def _encoded_master_sentence_list(self, text):
        '''Create master list of all sentences encoded to an integer in data set/corpus.'''
        for sentence in text:
            sent = [self.uniq_wrd_dict[w] for w in sentence]
            self.all_sentences_encoded.append(sent)

    def _count_uniq_words(self, text):
        '''Retrieve the occurrence of all words in the text_col for the entire data set/corpus.'''
        for sentence in text:
            for word in sentence:
                self.bag_of_words.append(word)
        # Sort words in descending order:
        self.word_counts = dict(Counter(self.bag_of_words).most_common())
        self.uniq_words = list(self.word_counts.keys())

    def _make_uniq_word_dict(self):
        '''Convert self.uniq_words into a dict() and create the reversed dict().'''
        self.n_words = len(self.uniq_words)
        # Sort words alphabetically instead of by count descending:
        self.uniq_words.sort()
        # self.uniq_wrd_dict = dict(sorted(zip(self.uniq_words, range(len_words)), key= lambda x: x[0])) # Not needed, below line works
        self.uniq_wrd_dict = dict(zip(self.uniq_words, range(self.n_words)))
        self.idx2word = dict(zip(range(self.n_words), self.uniq_words)) # dict(map(reversed, self.uniq_wrd_dict.items()))

    def _check_text_type(self, df):
        '''Check that self.text_col is congruous with expected data type.'''
        # Hacky way to check data type, but it works:
        content = df[self.text_col].iloc[0]
        if not isinstance(content, list):
            try:
                # Print messages only for debugging:
                # print(f'input data type for "df.{self.text_col}" == {type(content)}')
                df[self.text_col] = df[self.text_col].apply(lambda x: ast.literal_eval(x))
                # print(f'output data type for "df.{self.text_col}" == {type(df[self.text_col].iloc[0])}')
            except TypeError:
                print(f'The data type within {self.text_col} should either be str, list, or pd.Series; got {type(self.text_col)} instead.')

    def fit(self, X, y=None):
        '''Standard fit function.'''
        return self

    def transform(self, df):
        '''Create integer encoded sentences & create word2idx/idx2word dict()'s.'''
        self._check_text_type(df)
        # Create master list of sentences; useful only for testing
        # NOT NEEDED; df[self.text_col].apply(lambda x: self._master_sentence_list(x))
        # Create counter object to count word freqs
        df[self.text_col].apply(lambda x: self._count_uniq_words(x))
        # NOT NEEDED; Create the (context word, center word) pairs
        # df[self.text_col].apply(lambda x: self.word_neighbors(x))
        # Create unique word dictionary
        self._make_uniq_word_dict()
        # Create master list of encoded sentences, this is dependent on self._add_uniq_words() running prior
        df[self.text_col].apply(lambda x: self._encoded_master_sentence_list(x))
        # Return the object itself, this can cause issues if trying to re-use the pipeline on multiple dataframes
        # because returning self doesn't seem to differentiate between different dataframes. i.e. all data gets jumbled together
        return self

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    '''Average a single document of word vectors into one vector- a document vector!

    This class ingests text as a list of sentences, and maps each word to its word embedding vector. All word embedding
    vectors have the same dimension, so averaging them is not an issue.
    '''

    def __init__(self, word2idx, weight_mat,  text_col='ALLEGATION'):
        self.text_col = text_col
        self.all_sentences_encoded = []
        self.word2idx = word2idx
        self.weight_mat = weight_mat

    def vectorize_text(self, text):
        '''Convert list of list of words into one list of all word vectors.'''
        # return [word for sentence in text for word in sentence]
        mapped = [self.weight_mat[self.word2idx[word]] for sentence in text for word in sentence if
                  word in self.word2idx]        # np.nanmean allows mean to be computed if missing values are present
        return np.nanmean(mapped, axis=0)

    def _check_text_type(self, df):
        '''Check that self.text_col is congruous with expected data type.'''
        # Hacky way to check data type, but it works:
        content = df[self.text_col].iloc[0]
        if not isinstance(content, list):
            try:
                # Print messages only for debugging:
                # print(f'input data type for "df.{self.text_col}" == {type(content)}')
                df[self.text_col] = df[self.text_col].apply(lambda x: ast.literal_eval(x))
                # print(f'output data type for "df.{self.text_col}" == {type(df[self.text_col].iloc[0])}')
            except TypeError:
                print(
                    f'The data type within {self.text_col} should either be str, list, or pd.Series; got {type(self.text_col)} instead.')

    def fit(self, X, y=None):
        '''Standard fit function.'''
        return self

    def transform(self, X):
        '''Create a document vector based on the individual word vectors present in the SRSP text.'''
        self._check_text_type(X)
        # Can remove creation of column below, but keeping it now in case I want to use the doc vector downstream
        X['document_vector'] = X[self.text_col].apply(lambda x: self.vectorize_text(x))
        X.dropna(axis=0, subset=['document_vector'], inplace=True)
        # print(f"length of word vectors: {len(X['mapped_word_vectors'].values)}")
        values = X['document_vector'].values
        return np.stack(values)

class BrownCorpusReader():
    def __init__(self):
        self.sentences = {}
        self.words = {}
        self.n = -1
        try:
            self.corpus = nltk.corpus.brown
        except (ImportError, ModuleNotFoundError):
            print('Brown corpus has not been imported or you do not have the Brown corpus installed!')

    def get_sents(self, n):
        sent_list = []
        corpus = self.corpus
        for doc in corpus.fileids()[:n]:
            # print(doc)
            for sent in corpus.sents(doc):
                sent_list.append(sent)
                self.sentences[doc] = list(sent_list)

    def make_df(self, n=-1):
        '''Convert nltk's structured text into a 2 column dataframe.

        Columns include the fileid and the text of the file.

        Params
        ------
        n : int
            The number of files from the brown corpus to load. Default -1 will load all files.
        '''
        corpus = self.corpus
        ids = [file for file in corpus.fileids()[:n]]
        content = []
        for file in ids:
            words = list(corpus.words(file))
            content.append(' '.join(i for i in words))

        return pd.DataFrame({'files' : ids,
                             'text' : content}, columns=['files', 'text'])

    def fit(self, X, y=None):
        return self

    def transform(self, X=None, y=None, n=-1):
        '''Standard transform.'''
        return self.make_df(n=n)

class DummyTransformer(BaseEstimator, TransformerMixin):
    '''Dummy Transformer that does nothing :) !!!

    Purpose of this class is to simply return the X array untouched. An example use-case would be using this transformer
    in a sklearn.Pipeline where we want to scale the features of the X array/matrix, but would also like to compare results
    against NOT scaling the X input. The DummyTransformer() accomplishes this task.'''
    def fit(self, X, y=None):
        '''Standard fit function.'''
        return self

    def transform(self, X):
        return X

# Custom Estimator Classes
class KMeansEstimator(BaseEstimator, TransformerMixin):
    '''Kmeans class derived from sklearn.cluster.'''
    def __init__(self, k=25):
        # self.k_means=k_means
        self.k = k
        self.clusterer = KMeans(n_clusters=self.k, random_state=0)

    def fit(self, X, y=None):
        '''This can have some basic transformations on X. '''
        self.clusterer_ = clone(self.clusterer) # This is following sklearn convention with trailing '_'
        self.clusterer.fit(X)
        return self

    def transform(self, X):
        '''let's try return self to see if this gets rid of the AttributeError.'''
        return self

    def predict(self, X):
        return self.clusterer.predict(X)

    # def score(self):
    #     '''Is this needed???'''
    #     pass

class KMeansClusters(BaseEstimator, TransformerMixin):
    '''Directly from the book, this KMeans instance is from nltk instead of directly from sklearn.'''

    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """Fits k-means to one-ht encoded vectorize documents."""
        return self.model.cluster(documents, assign_clusters=True)

class ElbowPlot(BaseEstimator, TransformerMixin):
    '''Class for housing steps to acquire values for plotting Elbow plot for nltk.cluster.KMeansClusterer algorithm

    Params
    ------
    k_range : list/array
        This is the desired values of 'k' to be used in KMeans alg. This list or range should include ALL values of k
        that you wish to compute.
    df : pd.DataFrame()
        The data

    Attributes
    ----------
    self.wss : dict()
        The resulting dict() with K as keys and the inertia score as values. This is the data we will directly use to
        plot the elbow plot.

        {k: intertia score}
    '''

    def __init__(self, k_range, text_col=None, standalone=True):  # df,
        self.k_range = k_range
        self.standalone = standalone  # Really don't need this, as this should almost exclusively be standalone
        # self.df = df
        self.text_col = text_col
        # if not standalone: # add condition for either using an exterior pipeline OR just using internal pipe
        self.wss = dict(defaultdict())
        # self.model = Pipeline(steps=[
        #          ('kmeans', KMeansEstimator(k=10))  # {nltk_version : KMeansClusters, sklearn : KMeansEstimator}
        #     ])

    def _compute_centroid(self):
        # This will return the centroid of each cluster as an array
        return self.model.named_steps['kmeans'].model.means()

    def _compute_distance_to_centroid(self, centroid, df):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):

        for k in self.k_range:
            self.model = Pipeline(steps=[
                ('kmeans', KMeansEstimator(k=k))  # {nltk_version : KMeansClusters, sklearn : KMeansEstimator}
            ])
            # self.model['kmeans'].k = k
            self.model.fit_transform(X)
            self.wss[k] = round(self.model['kmeans'].clusterer.inertia_, 3)

        self.dynamic_bar_plot(self.wss, type=plt.plot,
                              title=f'Elbow Plot for k in [{min(self.k_range)}, {max(self.k_range)}]')
        return self.wss

    @staticmethod
    def dynamic_bar_plot(data, type=plt.bar, title=None, figsize=(15, 7), log=False, **kwargs):
        """data param must either be series object OR x,y values should be in a list Eg data=[ [x], [y] ]."""
        # handle series data. Eg: value_counts() format
        if isinstance(data, pd.Series):
            x, y = data.keys(), data.values
        # handle data in a list() format. Eg: [x,y]
        elif isinstance(data, dict):
            x, y = list(data.keys()), list(data.values())
        else:
            x, y = data[0], data[1]
        if log:
            y = np.log(y)
        # print(x, y )
        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=figsize)
            plt.xticks(rotation=90)  # ,**kwargs
            type(x, y, **kwargs)
            plt.title(title)

class HierarchicalClusters( BaseEstimator, TransformerMixin):

    def __init__(self):
        self.model = AgglomerativeClustering()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''This is an agglomerative model. Briefly, agglomerative models work from the 'bottom up'. Where every point is
         initially it's own cluster BUT is iteratively merged together up the hierarchy.'''

        clusters = self.model.fit_predict(X)
        self.y = self.model.labels_
        self.children = self.model.children_

        return clusters

class NaiveBayesClf (BaseEstimator, TransformerMixin):

    def __int__(self):
        self.model = MultinomialNB()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        xprime = X

class SklearnTopicModels(BaseEstimator, TransformerMixin):

    def __init__(self, txt_col='MEMINCIDENTREPORT', n_components=10):
        """
        n_components is the desired number of topics
        """
        self.n_components = n_components
        self.txt_col = txt_col
        self.model = Pipeline([
        ('norm', TextNormalizer(text_col=self.txt_col)),
        ('vect', CountVectorizer(input='content')), # Need to update this
        ('model', LatentDirichletAllocation(n_components=self.n_components)),
     ])

    def fit_transform(self, X, y=None):
        self.model.fit_transform(X)

        return self.model

    def get_topics(self, n=25):
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.feature_names
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[: -(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

    @staticmethod
    def print_topics(dictlike):
        for topic, terms in dictlike.items():
            print(f'topic: {topic}')
            print(terms)

class FbiTopicModels(SklearnTopicModels):
    '''Class inheriting from our SklearnTopicModels(BaseEstimator, TransformerMixin) base class.
    As such, this class contains a fit_transform() method, but not fit() or transform() as standalone funcs.

    Params
    ------
    include_normalizer : bool
        This is a convenience parameter in case the data has already been normalized from a previous step.
        If data has already been normalized, include_normalizer should be set to: False.
    n_components : int
        The desired number of topics to output
         '''

    def __init__(self, include_normalizer=True, txt_col='MEMINCIDENTREPORT', n_components=7):
        super().__init__()# n_components
        self.topic_labels = None
        self.txt_col = txt_col
        self.n_components = n_components
        self.max_df = .95
        self.min_df = 20
        self.random_state = 0
        if not include_normalizer:
            self.model = Pipeline([
                ('vect', TextCountVectorizer(text_col=self.txt_col, max_df=self.max_df, min_df=self.min_df)),
                ('model', LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)),
            ])
        # This is the default pipeline
        else:
            self.model = Pipeline([
            ('norm', TextNormalizer(text_col=self.txt_col)),
            ('vect', TextCountVectorizer(text_col=self.txt_col, max_df=self.max_df, min_df=self.min_df)),
            ('model', LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)),
         ])

    def fit_transform(self, X, y=None):
        xprime = X.copy() #maybe include this
        self.topic_labels = self.model.fit_transform(xprime)
        return self.model.fit_transform(xprime)
        # return self.model

    def assgn_topic_labels(self, df):

        topics = [i.argmax() for i in self.topic_labels]
        df['predicted_topic'] = topics

    @staticmethod
    def print_topics(dictlike):
        for topic, terms in dictlike.items():
            print(f'topic: {topic}')
            print(terms)

    def get_topics(self, n=25):
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.feature_names
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[: -(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

class TrainTestSplitOnly():
    '''Class with sole purpose of storing train/test indices within a dictionary. Implementation below which can be
    copied/pasted in your IDE:

        train_test = TrainTestSplitOnly()
        # If df is pd.DataFrame: make sure to use .iloc as shown below
        train_test.get_train_test_splits(df, df.iloc[:, -1], n_splits=4)
        # Access the indices of the splits by entering key = {'test', 'train'}; update i to an integer: {0 <= i < n_splits}
        split_dict = train_test.splits_dict[i]['test']
        # Check that the splits DO in fact evenly split the data by the supplied label
        df_labeled.iloc[testing_split_idx].label.value_counts()

    '''
    def __init__(self):
        self.n_splits = None
        self.splits_dict = defaultdict(lambda: defaultdict())
        # self.group_labels = None

    def cross_val_zipper(self):
        '''Zip values of train/test split for downstream pipeline GridSearchCV convenience.'''
        self.cross_val_zipped = []
        for key in self.splits_dict.keys():
            self.cross_val_zipped.append(list((self.splits_dict[key]['train'], self.splits_dict[key]['test'])))


    def get_train_test_splits(self, df, groups=None, n_splits=4):
        '''Returns the index of the train and test indices after splitting the data via StratifiedKFold() with n_splits.

        Params
        ------
        df : pd.Dataframe or np.array type
            This is the input data, which should be in a [n, d] shape. Can be either a dataframe or np array/matrix
        groups : list/array- like
            The groups are essentially the labels. The stratification of the data relies SOLELY on there being labels so
            that we can get equal representation of the labels in each train/test split.

            *NOTE* this should NOT simply be a unique list of all possible labels, this needs to be a list-like object with
            a value for every row in the data. len(groups) == len(df).
            In most cases -> groups = y
        '''
        self.n_splits = n_splits
        # Below IF applies if df is a pd.DataFrame
        if isinstance(df, pd.DataFrame):
            idx = len(df.columns) - 1
            X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        # Below ELSE clause handles if df is an np.array or matrix
        else:
            idx = df.shape[1] - 1
            X, y = df[:, 0:idx], df[:, -1]

        # In most cases groups should == y, which should be the last column in df i.e. (df.columns[-1])
        if not groups:
            groups = y
        # self.group_labels = y.unique()

        grp_stratified = StratifiedKFold(n_splits, random_state=101)
        grp_stratified.get_n_splits(X, y, groups)

        for i, (tr_idx, tst_idx) in enumerate(grp_stratified.split(X, y, groups=y)):
            # Below 2 lines not needed because we ONLY want the indices of train and test
            # X_train, X_test = X[tr_idx], X[tst_idx]
            # y_train, y_test = y[tr_idx], y[tst_idx]
            self.splits_dict[i] = {'train': tr_idx, 'test': tst_idx}

class TrainTestSplitWrapper():
    '''This is an outdated version and should not be used. Use TrainTestSplitOnly() instead!'''

    def __init__(self): #, split_type=None
        # self.split_type = split_type
        self.clf = None
        # Add a pipeline here?
        # self.model = Pipeline([])

    def confusion_matrix(self, **kwargs):
        """Plot confusion matrix for outputs. Need to implement log() of sums, otherwise values are too jumbled."""
        mat = confusion_matrix(self.best_tr_tst_dict['y_test_val'], self.best_tr_tst_dict['prediction'], **kwargs)
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
        plt.xlabel('predicted value')
        plt.ylabel('true value')

    def get_train_test_splits(self, df, groups, initial_run=True, n_splits=4):

        # Think I actually may need the isinstance() method BECAUSE if dataframe: I have to use to_numpy(); else: already np.array
        if isinstance(df, pd.DataFrame):
            idx = len(df.columns) - 1
            X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        # Else clause handles if df is an np.array or matrix
        else:
            idx = df.shape[1] - 1
            X, y = df[:, 0:idx], df[:, -1]
        # clf = MultinomialNB()
        clf = tree.DecisionTreeClassifier()
        grp_stratified = StratifiedKFold(n_splits, random_state=101)
        grp_stratified.get_n_splits(X, y, groups)
        best_score = 0
        split = 0
        for tr_idx, tst_idx in grp_stratified.split(X, y, groups):
            X_train, X_test = X[tr_idx], X[tst_idx]
            y_train, y_test = y[tr_idx], y[tst_idx]
            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            score = round(accuracy_score(y_test, prediction), 3)
            # Find me the best scoring train set... DO IT NOW!
            if score > best_score:
                best_score = score
                needed_keys = ['split', 'score', 'X_train_val', 'y_train_val', 'X_train_idx', 'X_test_val', 'y_test_val',
                               'X_test_idx', 'prediction']
                corresponding_vals = [split, score, X_train, y_train, tr_idx, X_test, y_test, tst_idx, prediction]
                if initial_run:
                    self.best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.clf = clf
                else:
                    self.iterate_best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.iterate_clf = clf
            split += 1

class PCAVisTranformer(BaseEstimator, TransformerMixin):
    '''Perform PCA to reduce data to a 2-d space for plotting.

    '''
    def __init__(self, n_components, text_col, data_name='Subset X', color_by=None):
        self.n_components = n_components
        self.text_col = text_col
        self.data_name = data_name
        self.model = PCA(n_components=self.n_components)
        self.color_by =color_by
        # self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.Xprime = np.stack(X[self.text_col].values)
        self.model.fit(self.Xprime)
        return self

    def transform(self, X):
        self.points = self.model.transform(self.Xprime)
        self.points_x, self.points_y = [i[0] for i in self.points], [i[1] for i in self.points]
        self._plot_data()
        return self.model.transform(self.Xprime)
        # Xprime = np.stack(X[self.text_col].values)

    @staticmethod
    def scatter_plotter(x, y, partition_by=None, title='title'):
        '''Standalone scatter plotting method.'''
        plt.subplots(figsize=(10, 7))
        legend_names = list(np.unique(partition_by))
        scatter = plt.scatter(x, y, c=partition_by)
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=legend_names,
                   title='Cluster')
        plt.title(f'{title}')

    def _plot_data(self):
        plt.subplots(figsize=(10, 7))
        if self.color_by:
            plt.scatter(self.points_x, self.points_y, c=self.color_by)
        plt.scatter(self.points_x, self.points_y)
        plt.title(f'{self.data_name} Document Vectors PCA-decomposed to {self.n_components}-d')


