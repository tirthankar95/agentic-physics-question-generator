import nltk 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from LLM.llm import get_response
from nltk import pos_tag
from typing import Literal

def is_noun(word):
    try:
        pos_tagged = pos_tag([word])
    except:
        nltk.download('averaged_perceptron_tagger_eng')
        pos_tagged = pos_tag([word])
    tag = pos_tagged[0][1]
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    return tag in noun_tags

def replace_words_wordnet(sentence, units):
    try:
        words = word_tokenize(sentence)
    except:
        nltk.download('punkt_tab')
        words = word_tokenize(sentence)
    synsets = []
    for word in words:
        try:
            synset_list = wordnet.synsets(word)
        except:
            nltk.download('wordnet')
            synset_list = wordnet.synsets(word)
        if synset_list and is_noun(word) and \
            (word not in units): synsets.append(synset_list[0].name().split('.')[0])
        else: synsets.append(word)
    return " ".join(synsets)

def replace_words_llm(sentence):
    instructions = """To increase the diversity and creativeness of the sentence change some words of the following sentence.
Do not provide any suggestions or add extra explanations, just output the final sentence.\n"""
    return get_response(instructions + sentence)
    
    
def replace_words(sentence, units, strategy: Literal['llm','wordnet'] = 'wordnet'):
    if strategy == "wordnet":
        return replace_words_wordnet(sentence, units)
    if strategy == "llm":
        return replace_words_llm(sentence)


if __name__ ==  '__main__':
    sen = "The dog died and went to heaven."
    print(f"[INP]: {sen}")
    print(f"[OUT]: {replace_words(sen, ['m/s', 'm'])}")