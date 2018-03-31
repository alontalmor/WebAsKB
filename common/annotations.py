from config import *
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from pycorenlp import StanfordCoreNLP

class Annotations:
    def __init__(self):
        self.st_caseless_NER = StanfordNERTagger("english.muc.7class.caseless.distsim.crf.ser.gz")
        self.nlp = StanfordCoreNLP(config.StanfordCoreNLP_Path)
        self.punc = list(set([",", "...", "-", ".", "'s", ":", "|", "?", "(", ";", ")", "'", "``", "..", "''"]))

    def annotate_question(self, question, annotators='tokenize,pos,lemma,ner,depparse'):
        question_dict = {}
        # removing the question mark
        question = question.replace('?', '')

        text = unicodedata.normalize('NFKD', question).encode('ascii', 'ignore').decode(encoding='UTF-8')

        output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,pos,lemma,ner,depparse',
            'outputFormat': 'json'
        })

        question_dict['word'] = [word['word'] for word in output['sentences'][0]['tokens']]
        question_dict['lemma'] = [word['lemma'] for word in output['sentences'][0]['tokens']]
        question_dict['ner'] = [word['ner'] for word in output['sentences'][0]['tokens']]

        question_dict['pos'] = [word['pos'] for word in output['sentences'][0]['tokens']]
        question_dict['is_stopword'] = list(pd.Series(question_dict['word']).isin(set(stopwords.words('english'))) * 1)

        # adding dependancies
        question_dict['question_dependencies'] = {}
        question_dict['question_dependencies']['basicDependencies'] = output['sentences'][0]['basicDependencies']
        question_dict['question_dependencies']['enhancedPlusPlusDependencies'] = output['sentences'][0]['enhancedPlusPlusDependencies']
        question_dict['question_dependencies']['enhancedDependencies'] = output['sentences'][0]['enhancedDependencies']

        return question_dict