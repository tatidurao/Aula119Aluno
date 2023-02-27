#Biblioteca de pré-processamento de dados de texto
import nltk

from nltk.stem import PorterStemmer
#classe é responsável por fornecer as palavras-tronco para as palavras dadas.
stemmer = PorterStemmer()

import json
#O pickle é a biblioteca Python que converte listas, dicionários e
#outros objetos em fluxos de zeros e uns. Isso será útil para
#armazenar dados de treinamento pré-processados.
import pickle
import numpy as np

nltk.download('punkt')

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

#função para anexar palavras-tronco
def get_stem_words(words, ignore_words):
    #palavra tronco
    stem_words = []
    for word in words:
        if word not in ignore_words:
            # steam(ver slide) para converter a palavra em sua palavra-tronco ou raiz.
            #transforma em palavra troco . lower() deixar minusculo
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Adicione todas as palavras dos padrões à lista
        #padrao
        for pattern in intent['patterns']:   
            #tokenizar, ou dar um codigo pra essas palavras         
            pattern_word = nltk.word_tokenize(pattern)     
            #adicionara os padroes tokenizados na lista de palavras       
            words.extend(pattern_word)  
            #palavra padrão e a tag                    
            word_tags_list.append((pattern_word, intent['tag']))
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        #criar lçista de palavra tronco para pegar a s plavras e excluie simbolos
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0]) 
print(classes)   

#Crie o corpus de palavras para o chatbot
#palçavras tronco e classe
def create_bot_corpus(stem_words, classes):
    #deixar as listas ordenadas
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))
    #criando arquivos que criar o conjunto de dados de treinamento.
    pickle.dump(stem_words, open('words.pkl','wb'))
    #“wb” significa escrita em modo binário. Assim, os dados armazenados 
    # de forma binaria 0 e 1
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags

#[000
# 000
# 000]