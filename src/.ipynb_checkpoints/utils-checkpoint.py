import os
import re
import math
from collections import Counter
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz


import pandas as pd
import numpy as np
import nltk
import gensim
from nltk.corpus import stopwords, brown
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, log_loss
from scipy.spatial.distance import cosine


def preprocess(text):
    """ Convert to lowercase and special characters, remove spaces, HTML tags, and punctuations marks 
    Args:
        text: Text to be converted. (str)
    Returns:
        text: Converted text. (str)
    """
    # Convert to lowercase and remove spaces
    text = str(text).lower().strip()

    # Replace certain special characters with their string equivalents  
    text = text.replace('$', ' dollar ')
    text = text.replace('€', ' euro ')
    text = text.replace('@', ' at ')
    text = text.replace('%', ' percent')

    # '[math]'
    text = text.replace('[math]', '')

    # Delete HTML tags
    text = BeautifulSoup(text,features="lxml")
    text = text.get_text()

    # Remove punctuations marks
    pattern = re.compile('\W')
    text = re.sub(pattern, ' ', text).strip()

    return text


def remove_stopwords(sentence):
    """ Remove stop words 
    Args:
        sentence: A string containing words. (str)
    Returns:
        result: A string containing no stop words. (str)
    """
    stopwords_list = stopwords.words('english')
    words = sentence.split(" ")
    filtered_sentence = [word for word in words if word not in stopwords_list]
    result = ' '.join([i for i in filtered_sentence if len(i) >= 2])

    return result

def preprocess_dataframe(df,target_cols,new_target_cols,directory_inter,df_filename):
    """ Applies preprocess, remove_stopwords, and loads/saves a csv file and returns a dataframe
    Args:
        df: A dataframe containing the questions to apply the preprocess and remove_stopwords on. (dataframe)
        target_cols: Names of columns for questions 1 and 2. (str list)
        new_target_cols: Name of columns for questions 1 and 2 after transformation. (str list)
        directory_inter: Directory to save the resulting dataframe. (str)
        df_filename: Name of the file to save the resulting dataframe. (str)
    Returns:
        df: A dataframe containing the questions after transformation. (dataframe)
    """
    if os.path.exists(os.path.join(directory_inter,df_filename)):
        df = pd.read_pickle(os.path.join(directory_inter,df_filename))
        return df
    
    for new_target_col, target_col in zip(new_target_cols,target_cols):
        df[new_target_col] = df[target_col].apply(preprocess).apply(remove_stopwords)
        
    df = df.dropna()
        
    df.to_pickle(os.path.join(directory_inter,df_filename))
    
    return df

def feature_engineering(df,questions_cols, questions_split_cols,directory_inter,df_filename):
    """ Creates multiple features given questions 1 and 2 and loads/saves and returns a dataframe that includes created features
    Args:
        df: A dataframe containing questions 1 and 2. (dataframe)
        questions_cols: Names of questions 1 and 2 columns. (str list)
        questions_split_cols: Names of questions 1 and columns after performing a split by space. (str list)
        directory_inter: Directory to save the resulting dataframe. (str)
        df_filename: Filename of dataframe containing created features. (str)
    Returns:
        df: A dataframe containing created features. (dataframe)
    """
    if os.path.exists(os.path.join(directory_inter,df_filename)):
        df = pd.read_pickle(os.path.join(directory_inter,df_filename))
        return df
    
    df[questions_split_cols[0]] = df.apply(lambda x: x[questions_cols[0]].split(),axis=1)
    df[questions_split_cols[1]] = df.apply(lambda x: x[questions_cols[1]].split(),axis=1)
    
    q1 = questions_split_cols[0]
    q2 = questions_split_cols[1]
    
    q1_og = questions_cols[0]
    q2_og = questions_cols[1]

    df['words_num_q1'] = df.apply(lambda x: len(x[q1]),axis=1)
    df['words_num_q2'] = df.apply(lambda x: len(x[q2]),axis=1)
    df['words_total'] = df['words_num_q1'] + df['words_num_q2']
    df['words_num_uniq_q1'] = df.apply(lambda x: len(set(x[q1])),axis=1)
    df['words_num_uniq_q2'] = df.apply(lambda x: len(set(x[q2])),axis=1)
    # words_num_uniq_total (with intersection between q1 and q2)
    df['words_num_uniq_total'] = df['words_num_uniq_q1'] + df['words_num_uniq_q2']
    df['words_intersection'] = df.apply(lambda x: len(set(x[q1]) & set(x[q2])),axis=1)
    df['average_words'] = df.apply(lambda x: (len(x[q1]) + len(x[q2]))/2,axis=1)
    df['standard_deviation_words'] = df.apply(lambda x: np.std([len(x[q1]), len(x[q2])]),axis=1)
    df['average_words_uniq'] = df.apply(lambda x: (len(set(x[q1])) + len(set(x[q2])))/2,axis=1)
    df['standard_deviation_words_uniq'] = df.apply(lambda x: np.std([len(set(x[q1])), len(set(x[q2]))]),axis=1)
    df['freq_word_q1'] = df.apply(lambda x: Counter(x[q1]).most_common(1)[0][1] if len(x[q1]) > 0 else 0,axis=1)
    df['freq_word_q2'] = df.apply(lambda x: Counter(x[q2]).most_common(1)[0][1] if len(x[q2]) > 0 else 0,axis=1)
    df['abs_diff'] = df.apply(lambda x: abs(len(x[q1]) - len(x[q2])),axis=1)
    df['abs_diff_uniq'] = df.apply(lambda x: abs(len(set(x[q1])) - len(set(x[q2]))),axis=1)
    df['fuzz_ratio'] = df.apply(lambda x: fuzz.QRatio(x[q1_og], x[q2_og]),axis=1)
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x[q1_og], x[q2_og]),axis=1)
    df['token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(x[q1_og], x[q2_og]),axis=1)
    df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x[q1_og], x[q2_og]),axis=1)
    
    df = df[(df['words_num_q1']!=0) | (df['words_num_q2']!=0)]
    
    integer_features = ['words_num_q1','words_num_q2','words_total','words_num_uniq_q1','words_num_uniq_q2','words_num_uniq_total','words_intersection',
                        'freq_word_q1','freq_word_q2','abs_diff','abs_diff_uniq','fuzz_ratio','fuzz_partial_ratio','token_sort_ratio','token_set_ratio']
    
    for feature in integer_features:
        df[feature] = pd.to_numeric(df[feature], downcast='float')
    
    df.to_pickle(os.path.join(directory_inter,df_filename))
    
    return df  
    
    
def tfidf_vectorizer(questions):
    """ Calculates TF-IDF dictionary for all words in list questions
    Args:
        questions: A list containing questions in train data. (str list)
    Returns:
        vectorizer: A TfidfVectorizer() fitted on questions.
        idf: A dictionary with keys as words, and values as idf weights. (dict)
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(questions)
    
    # Map terms to their corresponding idf values
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    return vectorizer, idf


def weighted_unweighted_question(model,model_vocab,weighted,idf,q,model_dim):
    """ Calculates a vector representation based on an embedding model and TF-IDF weight or not
    Args:
        model: An initialized Word2Vec or GloVe pretrained model.
        model_vocab: Words in model.
        weighted: True if weighted TF-IDF, False otherwise. (boolean)
        idf: A dictionary with keys as words, and values as idf weights. (dict)
        q: A question in a list format.
        model_dim: Model vectors dimension. (int)
    Returns:
        vec_q: A vector based on an embedding model. (np array)
    """    
    vec_q = np.zeros((model_dim))
        
    if weighted:
        for word in q:
            # Multiply word vector with IDF weight, and add it to the mean vector
            vec_q += model[word] * idf[str(word)] if (word in idf) and (word in model_vocab) else 0

        # Divide the mean vector by the number of words in question
        vec_q /= len(q)
    else:
        for word in q:
            # Add word1 vector to vector1
            vec_q += model[word] if (word in model_vocab) else 0
    
    return vec_q
    

def weighted_unweighted_embedding(model_embedding,model_type,weighted,idf,q1,q2,model_dim):
    """ Given a model (Word2Vec or GloVe) it calculates similarity between two questions based on unweighted or weighted TF-IDF
    Args:
        model_embedding: An initialized Word2Vec or GloVe pretrained model.
        model_type: 'word2vec' or 'glove'. (str)
        weighted: True if weighted TF-IDF, False otherwise. (boolean)
        idf: A dictionary with keys as words, and values as idf weights. (dict)
        q1: Question 1 in a list. (str)
        q2: Question 2 in a list. (str)
        model_dim: model vectors dimension. (int)
    Returns:
        similarity: Cosine similarity of two questions based on a model and weighted or not. (float)
    """
    if model_type == 'word2vec':
        model = model_embedding.wv
        model_vocab = model.key_to_index       
    elif model_type == 'glove':
        model = model_embedding
        model_vocab = model
        
        
    if weighted:
        vec_q1 = weighted_unweighted_question(model,model_vocab,weighted,idf,q1,model_dim)
        vec_q2 = weighted_unweighted_question(model,model_vocab,weighted,idf,q2,model_dim)
    else:
        vec_q1 = weighted_unweighted_question(model,model_vocab,weighted,idf,q1,model_dim)
        vec_q2 = weighted_unweighted_question(model,model_vocab,weighted,idf,q2,model_dim)
        
    similarity = 1 - cosine(vec_q1,vec_q2)
    
    return similarity if not math.isnan(similarity) else 0

def w2v_glove_similarity(df,idf,model_w2v,model_glove,w2v_dim,glove_dim,directory_inter,df_full_features):
    """ Calculates unweighted and weighted Tf-IDF for both Word2Vec and GloVe models and loads/saves the result in a dataframe and returns it
    Args:
        df: A dataframe containing all features, and similarity scores to be written in it. (dataframe)
        idf: A dictionary with keys as words, and values as idf weights. (dict)
        model_w2v: An initialized Word2Vec pretrained model.
        model_glove: An initialized GloVe pretrained model.
        w2v_dim: Word2Vec vectors dimension. (int)
        glove_dim: GloVe vectors dimension. (int)
        directory_inter: Directory to save the resulting dataframe. (str)
        df_full_features: Filename of dataframe containing all features. (str)
    Returns:
        df: A dataframe containing all features. (dataframe)
    """
    
    if os.path.exists(os.path.join(directory_inter,df_full_features)):
        df = pd.read_pickle(os.path.join(directory_inter,df_full_features))
        return df
    
    df['similarity_w2v'] = df.apply(lambda x: weighted_unweighted_embedding(model_w2v,
                                                                            'word2vec',
                                                                            True,
                                                                            idf,
                                                                            x['q1_split'],
                                                                            x['q2_split'],
                                                                            w2v_dim),axis=1)
    
    df['similarity_unweighted_w2v'] = df.apply(lambda x: weighted_unweighted_embedding(model_w2v,
                                                                                       'word2vec',
                                                                                       False,
                                                                                       idf,
                                                                                       x['q1_split'],
                                                                                       x['q2_split'],
                                                                                       w2v_dim),axis=1)
    
    df['similarity_glove'] = df.apply(lambda x: weighted_unweighted_embedding(model_glove,
                                                                              'glove',
                                                                              True,
                                                                              idf,
                                                                              x['q1_split'],
                                                                              x['q2_split'],
                                                                              glove_dim),axis=1)
    
    df['similarity_unweighted_glove'] = df.apply(lambda x: weighted_unweighted_embedding(model_glove,
                                                                                         'glove',
                                                                                         False,
                                                                                         idf,
                                                                                         x['q1_split'],
                                                                                         x['q2_split'],
                                                                                         glove_dim),axis=1)
    
    df.to_pickle(os.path.join(directory_inter,df_full_features))
    
    return df


def scale(df_data,X_train,X_test,word_embeddings_similarities,features,merge_index,merge_how):
    """ Scale features based on train data only 
    Args:
        df_data: A dataframe containing all features and index. (dataframe)
        X_train: A dataframe containing train data and index. (dataframe)
        X_test: A dataframe containing test data and index. (dataframe)
        word_embeddings_similarities: A list of similarity features generated from Word2Vec and GloVe models. (str list)
        features: A list of features. (str list)
        merge_index: A merge index to be used to merge on. (str)
        merge_how: Method of merging. (str)
    Returns:
        X_train_final: A dataframe containing train data and similarity features generated from Word2Vec and GloVe models. (dataframe)
        X_test_final: A dataframe containing test data and similarity features generated from Word2Vec and GloVe models. (dataframe)
    """   
    X_train_merged = pd.merge(X_train,df_data[[merge_index]+word_embeddings_similarities],on=[merge_index,merge_index],how=merge_how)
    X_test_merged = pd.merge(X_test,df_data[[merge_index]+word_embeddings_similarities],on=[merge_index,merge_index],how=merge_how)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train_merged[features])
    X_train_scaled = pd.DataFrame(data = X_train_scaled, columns = [col+'_scaled' for col in features])
    X_test_scaled = sc.transform(X_test_merged[features])
    X_test_scaled = pd.DataFrame(data = X_test_scaled, columns = [col+'_scaled' for col in features])
    X_train_final = pd.concat([X_train_merged,X_train_scaled],axis=1)
    X_test_final = pd.concat([X_test_merged,X_test_scaled],axis=1)
    
    return X_train_final , X_test_final

def display_results(model,model_name,model_predictions,X_test,y_test,test_results,train_time): 
    """ Given a model, it prints model name, accuracy, balanced accuracy, matthews correlation coefficient, logloss, and 
        training time, and it saves them in datframe test_results.
    Args:
        model: A model to output its results on certain metrics. (model)
        model_name: Model name. (str)
        model_predictions: Predictions of the model on X_test. 
        X_test: Features to test on. (dataframe)
        y_test: Label for duplicate questions to test on. (dataframe) 
        test_results: A dataframe to save the results on certain metrics for this model. (dataframe)
        train_time: Training time for this model. (float)
    Returns:
        test_results: A dataframe to save the results on certain metrics for this model. (dataframe)
    """    
    if np.all((model_predictions==0) | (model_predictions==1)):
        accuracy = accuracy_score(y_test, model_predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, model_predictions)
        mcc = matthews_corrcoef(y_test, model_predictions)
        log_loss_func = log_loss(y_test, model.predict_proba(X_test))
        
    else:
        model_preds = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, model_preds)
        balanced_accuracy = balanced_accuracy_score(y_test, model_preds)
        mcc = matthews_corrcoef(y_test, model_preds)
        log_loss_func = log_loss(y_test, model_predictions)       

        
    print('Accuracy = {} \nBalanced Accuracy = {} \nMCC = {} \nlogloss = {} \ntraining time = {}'.format(accuracy, 
                                                                                                         balanced_accuracy, 
                                                                                                         mcc,
                                                                                                         log_loss_func,
                                                                                                         train_time))
    
    test_results.loc[len(test_results)] = [model_name,accuracy,balanced_accuracy,mcc,log_loss_func,train_time]
    
    return test_results

def load_w2v_model(brown_train_size,questions_list,w2v_dim,directory_inter,w2v_brown_filename):
    """ Loads/downloads a Word2Vec model and trains it on Brown dataset and questions_list dataset 
    Args:
        brown_train_size: Number of elements in Brown dataset. (int)
        questions_lists: A list of questions after splitting. (str list)
        w2v_dim: Dimension of Word2Vec vectors. (int)
        directory_inter: Directory to save the resulting dataframe. (str)
        w2v_brown_filename: Filename of a trained Word2Vec model. (str)
    Returns:
        model_w2v: A trained Word2vec model. (model)
    """
    if os.path.exists(os.path.join(directory_inter,w2v_brown_filename)):
        model_w2v = gensim.models.Word2Vec.load(os.path.join(directory_inter,w2v_brown_filename))
        return model_w2v
    
    nltk.download('brown')
    train_set = brown.sents()[:brown_train_size]
    model_w2v = gensim.models.Word2Vec(train_set + questions_list, vector_size = w2v_dim)
    
    # Save the word2vec model
    model_w2v.save(os.path.join(directory_inter,w2v_brown_filename))
    
    return model_w2v

def load_glove_model(directory_glove,glove_6b_filename,glove_temp_filename,directory_inter,glove_model_filename):
    """ Loads/downloads a GloVe model and trains it on a 6B tokens 400K vocab dataset
    Args:
        directory_glove: Directory to fetch GloVe model from. (str)
        glove_6b_filename: Name of the 6B tokens 400K vocab dataset file. (str)
        glove_temp_filename: Name of template file for the 6B tokens 400K vocab dataset file. (str)
        directory_inter: Directory to save the resulting dataframe. (str)
        glove_model_filename: Filename of a trained GloVe model. (str)
    Returns:
        model_glove: A trained GloVe model. (model) 
    """
    
    if os.path.exists(os.path.join(directory_inter,glove_model_filename)):
        model_glove = gensim.models.KeyedVectors.load(os.path.join(directory_inter,glove_model_filename))
        return model_glove
    
    word2vec_glove_file = get_tmpfile(glove_temp_filename)
    glove2word2vec(os.path.join(directory_glove,glove_6b_filename), word2vec_glove_file)
    model_glove = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    
    # Save the glove model
    model_glove.save(os.path.join(directory_inter,glove_model_filename))
    
    return model_glove