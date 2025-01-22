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
      
    return stem_words

