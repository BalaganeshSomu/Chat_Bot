    #---------------------------------------------------------#
    ### CHATBOT WITH TFIDF VECTORIZER AND COSINE SIMILARITY ###
    ### By: Balaganesh Somu                                 ###
    #---------------------------------------------------------#

import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from math import log10
from nltk import word_tokenize

def read_dataframe():
    '''
        ENTER THE LOCATION WHERE YOUR EXCEL/CSV/JSON/TXT
        FILE HAS BEEN STORED HERE
        The ffill is used to fill the empty rows with previous values
    '''
    
    df=pd.read_excel('C:\\Users\\Balaganesh\\Desktop\\projects\\ML\\chatbot\\responses.xlsx')
    df.ffill(axis = 0,inplace = True)
    return (df)

    #===========================================================#
    ###                  COSINE SIMILARITY                    ###
    #===========================================================#
class cosine_similarity:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        
    def calculate(self):
        ''' FORMULA FOR COSINE SIMILARITY
            dot(d1 , d2) / ||d1|| ||d2||
        '''
        numerator = sum([i*j for i,j in zip(self.X,self.Y)])
        denominator = sum([pow(i,2)*pow(j,2) for i,j in zip(self.X,self.Y)])
        if denominator == 0:
            return numerator
        return numerator/denominator

    #===========================================================#
    #####                TFIDF VECTORIZER                   #####
    #===========================================================#
class tfidf_vectorizer:
    
    def __init__(self,question_series):
        self.question_series = question_series
        
    def special_characters():
        '''
            A dictionary with all the special charaters as keys
            and None as values .This would be used for removing
            the special characters in the data
            RETURNS : DICTIONARY {ASCII(SPECIAL CHAR) : None}
        '''
        special_characters = dict((ord(i),None) for i in punctuation)
        return special_characters

    def stop_words():
        '''
            For efficient working all the stop words in english are
            removed from the data (e.g,)should, would......
            -- Apostrophes should be removed --
            RETURNS : STOP_WORDS
        '''
        punctuation = {ord("'"):None}
        stop_words = set(i.translate(punctuation) for i in stopwords.words('english'))
        return stop_words
    
    def remove_stopwords(text):
        '''
        Removes the stop words
        '''
        stop_words = tfidf_vectorizer.stop_words()
        return set(str(text).split(' '))-stop_words
    
    def normalize(self):
        '''
            Converts the string to lower case , remover special
            characters
        '''
        special_characters = tfidf_vectorizer.special_characters()
        series = self.question_series
        series = series.str.lower()
        series = series.str.translate(special_characters)
        return series

    def preprocess(self):
        '''
            CORPUS      : A list of all the questions
            TOKENS      : Consists of all the unique words
                          (after removing stopwords)
                          Would be used as columns for DataFrame
            WORDS_COUNT : A dictionary consisting of all unique
                          words as keys and number of times they
                          appear as values
            QUES_DF     : A DataFrame with normalized questions as one
                          column and their tokens (after removing stop
                          words) as another columns

            RETURNS required parameters for tfidf calculation
        '''
        corpus = list(self.question_series)
        ques_df = pd.DataFrame(self.normalize(),
                             columns=['Questions'])
        ques_df['Tokens'] = ques_df['Questions'].apply(tfidf_vectorizer.remove_stopwords)
        tokens = sorted(list(set([j for i in list(ques_df['Tokens']) for j in i])))
        questions = list(self.question_series)
        words = list(ques_df['Tokens'])
        all_words = [j for i in corpus for j in word_tokenize(str(i))]
        word_count = dict((i,all_words.count(i)) for i in tokens)
        return (ques_df,tokens,questions,words,word_count)

    def tfidf(self):
        '''
        Gets the ques_df, calculates the tfidf score for each token
        Return 2 DataFrames . One with tfidf-score for dataset
        other with tfidf score for question
        '''
        ques_df,tokens,questions,words,word_count = self.preprocess()
        N=len(self.question_series)
        def formula_tfidf(i,j):
            #   TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY
            #   tf-idf(t, d) = tf(t, d) * idf(t)
            #   idf(t) = log [ n / df(t) ] + 1
            #   where , t - term , d - document
            denominator=len(str(questions[j]).split(' '))
            numerator=list(words[j]).count(i)
            tf=numerator/denominator
            n=word_count[i]
            idf=1+log10((N+1)/(n+1))
            return tf*idf
        df_tfidf=pd.DataFrame([[formula_tfidf(i,j) for i in tokens] for j in range(len(words))],columns=tokens)
        Y=pd.DataFrame(df_tfidf.iloc[-1])
        X=df_tfidf[:-1]
        return (X,Y)


def main():
    print("Hi there ! I'm Roy. Would you like to chat with me?")
    print("Type EXIT if you wanna exit")

    df = read_dataframe()
    question_series = df['Context']
    
    while(True):
        question = input('YOU:')
        if question == 'EXIT':
            print('ROY : Adios! :(')
            break
        else:
            '''
                Get the question --> Find Tfidf --> Find cosine similarity --> Return answer 
            '''
            question_series = question_series.append(pd.Series(question),ignore_index=True)
            X,Y = tfidf_vectorizer(question_series).tfidf()
            C = [cosine_similarity(list(Y.values),list(X.iloc[i].values)).calculate() for i in range(len(X))]
            if max(C)==0:
                print("ROY : Hmm... I don't get it")
            else:
                print('ROY :'+str(df['Text Response'].iloc[C.index(max(C))]))

if __name__=='__main__':
    main()
