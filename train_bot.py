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

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

#função para anexar palavras-tronco
def get_stem_words(words, ignore_words):
    #
    stem_words = []
    for word in words:
        if word not in ignore_words:
            # 
            #
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words
    
#Repetidor para acessar os dados de treinamento    
for intent in intents['intents']:
    
        # 
        for pattern in intent['patterns']:   
                     
            pattern_word = nltk.word_tokenize(pattern)     
            #       
            words.extend(pattern_word)  
            #                 
            word_tags_list.append((pattern_word, intent['tag']))
        # 
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        #
            stem_words = get_stem_words(words, ignore_words)

#resultado
print("Palvras estimizadas:", stem_words)
print("Listas de palvras: ", word_tags_list[2]) 
print("classes: ",classes)   

#Crie o corpus de palavras para o chatbot
#
def create_bot_corpus(stem_words, classes):
    #
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))
    #
    pickle.dump(stem_words, open('words.pkl','wb'))
    #
    # 
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

#chamando e armazenando o retorno dos dados da função
stem_words, classes = create_bot_corpus(stem_words,classes)  

print("palavras stimizadas: ",stem_words)
print("classes: ", classes)

#listas a serem usadas no restante do modelo
training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags

# Crie um saco de palavras e o labels_encoding

# Crie os dados de treinamento

