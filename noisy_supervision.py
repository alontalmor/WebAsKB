from config import *
import pandas as pd
import numpy as np
import json
from common.annotations import Annotations
from common.embeddings import embeddings
from nltk.corpus import stopwords

class NoisySupervision():
    def __init__(self):

        # load word embeddings (a sample of GloVe 300)
        embed = embeddings()
        embed.load_vocabulary_word_vectors(config.glove_300d_sample,'glove.sample.300d.txt',300)
        self.wordvecs = embed.wordvecs
        self.load_data()
        self.annotate = Annotations()

    def load_data(self):
        # loading webcomplexquestions
        with open(config.complexwebquestions_dir + 'ComplexWebQuestions_' + config.EVALUATION_SET + '.json') as f:
            questions = json.load(f)
        print(len(questions))
        print(pd.DataFrame(questions)['compositionality_type'].value_counts())

        # aliases version
        compWebQ = pd.DataFrame([{'ID':question['ID'],'question':question['question'],'webqsp_question':question['webqsp_question'], \
            'machine_question':question['machine_question'],'comp':question['compositionality_type'], \
            'answers':[answer['answer'] for answer in question['answers']]} for question in questions])
        print(compWebQ['comp'].value_counts())

        self.compWebQ = compWebQ.to_dict(orient="rows")

    # calculates the similarity matrix A
    # where Aij is the similarity between token i in the MG question and token j in the NL question.
    # Similarity is 1 if lemmas match, or cosine similarity according to GloVe embeddings
    # (Pennington et al., 2014), when above a threshold, and 0 otherwise.
    def calc_similarity_mat(self, question):
        question['question'] = question['question'].replace('?', '').replace('.', '')
        question['machine_question'] = question['machine_question'].replace('?', '').replace('.', '')

        annotations = self.annotate.annotate_question(question['question'])
        machine_annotations = self.annotate.annotate_question(question['machine_question'], annotators='tokenize,pos,lemma')
        webqsp_annotations = self.annotate.annotate_question(question['webqsp_question'], annotators='tokenize')
        dep_str = ''

        for term1 in annotations['question_dependencies']['basicDependencies']:
            dep_str += term1[u'dep'].replace(' ', '_') + ' '

        dep_str = ' '.join(set(dep_str.split(' ')))
        question['rephrased_pos'] = annotations['pos']
        question['rephrased_tokens'] = annotations['word']
        question['machine_tokens'] = machine_annotations['word']
        question['webqsp_tokens'] = webqsp_annotations['word']
        question['rephrased_lemma'] = annotations['lemma']
        question['machine_lemma'] = machine_annotations['lemma']
        question['dep_str'] = dep_str
        question['annotations'] = annotations['question_dependencies']['basicDependencies']
        question['sorted_annotations'] = pd.DataFrame(
            annotations['question_dependencies']['basicDependencies']).sort_values(by='dependent').to_dict(orient='rows')

        # calculating original split point
        org_q_vec = question['webqsp_tokens']
        machine_q_vec = question['machine_tokens']
        org_q_offset = 0

        for word in machine_q_vec:
            if org_q_offset < len(org_q_vec) and org_q_vec[org_q_offset] == word:
                org_q_offset += 1
            else:
                break

        # adding split_point2 for composition
        if question['comp'] == 'composition':
            org_q_offset2 = len(machine_q_vec) - 1
            for word in org_q_vec[::-1]:
                if org_q_offset2 > 0 and machine_q_vec[org_q_offset2] == word:
                    org_q_offset2 -= 1
                else:
                    break
            if org_q_offset2 != len(machine_q_vec) - 1:
                question['split_point2'] = org_q_offset2
            else:
                question['split_point2'] = org_q_offset2

            question['machine_comp_internal'] = ' '.join(
                question['machine_tokens'][org_q_offset:question['split_point2'] + 1])

        question['split_point'] = org_q_offset
        if question['split_point'] == 0:
            question['split_point'] = 1

        question_words = [word.lower() for word in question['rephrased_tokens']]
        span_words = [word.lower() for word in question['machine_tokens']]

        org_q_offset = 0
        new_part = []
        for word in question['machine_tokens']:
            if org_q_offset < len(question['webqsp_tokens']) and question['webqsp_tokens'][org_q_offset] == word:
                org_q_offset += 1
            else:
                new_part.append(word)

        question['split_point'] = org_q_offset
        question['new_part'] = ' '.join(new_part)

        q_vec_list = []
        for word in question_words:
            if word in self.wordvecs:
                q_vec_list.append(self.wordvecs[word])
            else:
                q_vec_list.append(np.zeros([300], dtype='float'))
        qvecs = np.asarray(q_vec_list, dtype='float')

        s_vec_list = []
        for word in span_words:
            if word in self.wordvecs:
                s_vec_list.append(self.wordvecs[word])
            else:
                s_vec_list.append(np.zeros([300], dtype='float'))
        svecs = np.asarray(s_vec_list, dtype='float')

        # compute the similarity between every question word vector a in A, and span word vector b in B
        # here we compute the similarity with cosine distance
        qvecs_norm = np.linalg.norm(qvecs, axis=1)
        svecs_norm = np.linalg.norm(svecs, axis=1)

        similarity = np.divide(np.dot(qvecs, svecs.T), np.outer(qvecs_norm, svecs_norm))
        similarity = similarity.transpose()
        similarity[np.isnan(similarity)] = 0
        similarity[similarity < 0.37] = 0

        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                if question['rephrased_lemma'][j] == question['machine_lemma'][i]:
                    similarity[i, j] = 1

        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                if question_words[j] == span_words[i]:
                    similarity[i, j] = 1

        # similarity "weights"
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                if question_words[j] in stopwords.words('english') or span_words[i] in stopwords.words('english'):
                    similarity[i, j] = similarity[i, j] * 0.5

                if question['rephrased_pos'][j].find("NN") == 0:
                    similarity[i, j] = similarity[i, j] * 1.3

        enhanced_similarity = similarity.copy()
        for i in range(similarity.shape[0] - 1):
            for j in range(similarity.shape[1] - 1):
                enhanced_similarity[i + 1, j + 1] += similarity[i, j] * 0.3
                enhanced_similarity[i, j] += similarity[i + 1, j + 1] * 0.3
        similarity = enhanced_similarity

        return question, similarity, annotations, question_words, span_words

    # see Generating noisy supervision in Talmor and Berant 2018 https://arxiv.org/abs/1803.06643
    # (the noisy supervision is a heuristic and comes with some noisy code :)
    def gen_noisy_supervision(self):
        qind = 0
        num_q_to_proc = len(self.compWebQ)
        for question in self.compWebQ[0:num_q_to_proc]:

            # print question
            qind += 1
            if qind % 100 == 0:
                print(qind)

            if question['comp'] is None or question['comp'] in ['comparative', 'superlative']:
                annotations = self.annotate.annotate_question(question['question'])
                question['sorted_annotations'] = pd.DataFrame(
                    annotations['question_dependencies']['basicDependencies']).sort_values(by='dependent').to_dict(
                    orient='rows')

                continue

            # For every question, a similarity matrix A is constructed,
            # where Aij is the similarity between token i in the MG question and token j in the NL question.
            # Similarity is 1 if lemmas match, or cosine similarity according to GloVe embeddings
            # (Pennington et al., 2014), when above a threshold, and 0 otherwise.
            question, similarity, annotations, question_words, span_words = self.calc_similarity_mat(question)

            if question['split_point'] == 0:
                question['split_point'] = 1

            question['flip_rephrase'] = 0
            if question['comp'] == 'conjunction':
                annotations_dict = [x['dep'] for x in \
                                    pd.DataFrame(annotations['question_dependencies']['basicDependencies']).sort_values(
                                        by='dependent').to_dict(orient='rows')]
                diff1 = []
                diff2 = []
                for j in range(0, similarity.shape[1]):
                    if j < 3 or j > similarity.shape[1] - 2:
                        diff1.append(0)
                        diff2.append(0)
                    else:
                        diff1.append(np.amax(similarity[0:question['split_point'], 0:j], axis=1).mean() + np.amax(
                            similarity[question['split_point']:, j:], axis=1).mean())
                        diff2.append(np.amax(similarity[question['split_point']:, 0:j], axis=1).mean() + np.amax(
                            similarity[0:question['split_point'], j:], axis=1).mean())

                if np.sum(diff1) > np.sum(diff2):
                    if len(diff1) == 0:
                        continue
                    Diff = diff1
                    question['flip_rephrase'] = 0
                else:
                    if len(diff2) == 0:
                        continue
                    Diff = diff2
                    question['flip_rephrase'] = 1

                # enhancing better split points:
                if 'that' in question['rephrased_tokens']:
                    Diff[question['rephrased_tokens'].index('that')] += 0.1
                if 'and' in question['rephrased_tokens']:
                    Diff[question['rephrased_tokens'].index('and')] += 0.1
                if 'which' in question['rephrased_tokens']:
                    Diff[question['rephrased_tokens'].index('which')] += 0.1
                if 'has' in question['rephrased_tokens']:
                    Diff[question['rephrased_tokens'].index('has')] += 0.05
                if 'is' in question['rephrased_tokens']:
                    Diff[question['rephrased_tokens'].index('is')] += 0.05

                question['p1'] = np.argmax(Diff)

                split_part1 = question['rephrased_tokens'][0:question['p1']]
                split_part2 = question['rephrased_tokens'][question['p1']:]

                question['split_part1'] = ' '.join(split_part1)
                question['split_part2'] = ' '.join(split_part2)

                question['p2'] = None

                # Det + nsubj case
                basicDep = pd.DataFrame(annotations['question_dependencies']['basicDependencies']).sort_values(
                    by='dependent').reset_index(drop=True)
                if basicDep.iloc[0]['dep'] == "det":
                    question['split_part2'] = \
                        basicDep.loc[basicDep['dependent'] == basicDep.iloc[0]['governor'], 'dependentGloss'].iloc[0] + ' ' + \
                        question['split_part2']
                    question['p2'] = int(
                        basicDep.loc[basicDep['dependent'] == basicDep.iloc[0]['governor'], 'dependentGloss'].index[0])
                else:
                    question['p2'] = 0

                question['max_diff'] = np.max(Diff)
                question['machine_comp_internal'] = ''

            else:
                if question['split_point2'] <= question['split_point']:
                    print('found error in split point 2')
                    question['split_point2'] = question['split_point'] = 1

                annotations['question_dependencies']['basicDependencies'] = \
                    pd.DataFrame(annotations['question_dependencies']['basicDependencies']).sort_values(by='dependent').to_dict(
                        orient='rows')
                Diff = np.zeros((similarity.shape[1], similarity.shape[1]))
                Diff_struct = {}
                for start in range(0, similarity.shape[1] - 2):
                    for end in range(start + 2, similarity.shape[1]):
                        vec = []
                        if start > 0:
                            vec += list(np.amax(similarity[0:question['split_point'], 0:start], axis=0))
                        if start > 0 and question['split_point2'] + 1 < similarity.shape[0]:
                            vec += list(np.amax(similarity[question['split_point2'] + 1:, 0:start], axis=0))

                        Diff[start, end] += np.amax(
                            similarity[question['split_point']:question['split_point2'] + 1, start:end + 1], axis=0).sum()
                        if end < similarity.shape[1] - 1:
                            vec += list(np.amax(similarity[0:question['split_point']:, end + 1:], axis=0))
                        if end < similarity.shape[1] - 1 and question['split_point2'] + 1 < similarity.shape[0]:
                            vec += list(np.amax(similarity[question['split_point2'] + 1:, end + 1:], axis=0))

                        if len(vec) > 0:
                            Diff[start, end] += sum(vec)

                        Diff_struct[str(start) + '_' + str(end)] = \
                            {'vec': vec, 'internal_vec': list(
                                np.amax(similarity[question['split_point']:question['split_point2'] + 1, start:end + 1],
                                        axis=0)), \
                             'diff': Diff[start, end], 'internal': ' '.join(question['rephrased_tokens'][start:end + 1])}

                max_inds = list(np.unravel_index(Diff.argmax(), Diff.shape))

                ##################################
                # Rule based refinements

                # refining the max inds
                while max_inds[1] < Diff.shape[1] - 1:
                    if Diff[max_inds[0], max_inds[1]] == Diff[max_inds[0], max_inds[1] + 1]:
                        max_inds[1] += 1
                    else:
                        break

                if Diff[max_inds[0] + 1, max_inds[1]] + 0.1 > Diff[max_inds[0], max_inds[1]]:
                    max_inds[0] += 1

                if annotations['question_dependencies']['basicDependencies'][max_inds[0]]['governorGloss'] == 'ROOT':
                    max_inds[0] += 1
                if annotations['question_dependencies']['basicDependencies'][max_inds[0] + 1]['governorGloss'] == 'ROOT':
                    max_inds[0] += 2
                if annotations['question_dependencies']['basicDependencies'][max_inds[0]]['dep'] == 'case':
                    max_inds[0] += 1

                    # "the" is usually part of the internal part
                if max_inds[0] > 0 and question['rephrased_tokens'][max_inds[0] - 1].lower() == 'the':
                    max_inds[0] -= 1

                question['p1'] = max_inds[0]
                question['p2'] = max_inds[1]
                question['max_diff'] = Diff.max()

                question['split_part1'] = ' '.join(question['rephrased_tokens'][max_inds[0]:max_inds[1] + 1])
                question['split_part2'] = ''
                if max_inds[0] > 0:
                    question['split_part2'] += ' '.join(question['rephrased_tokens'][0:max_inds[0]])
                question['split_part2'] += ' %composition '
                if max_inds[1] + 1 < len(question['rephrased_tokens']):
                    question['split_part2'] += ' '.join(question['rephrased_tokens'][max_inds[1] + 1:])
                question['split_part2'] = question['split_part2'].strip()

        out = pd.DataFrame(self.compWebQ[0:num_q_to_proc])[
            ['ID', 'comp', 'p1', 'p2', 'flip_rephrase', 'split_part1', 'machine_comp_internal', 'split_part2', 'question',
             'machine_question', 'answers', 'sorted_annotations', 'max_diff']]

        with open(config.noisy_supervision_dir + config.EVALUATION_SET + '.json', 'w') as outfile:
            json.dump(out.to_dict(orient="rows"), outfile, sort_keys=True, indent=4)

if __name__ == "__main__":
    noisy_sup = NoisySupervision()
    noisy_sup.gen_noisy_supervision()

    # testing
    with open('Data/noisy_supervision/dev.json', 'r') as outfile:
        split_dataset1 = pd.DataFrame(json.load(outfile))[0:100]
    with open('../WebKB/output/SP0.3_ComplexWebQuestions_dev.json', 'r') as outfile:
        split_dataset2 = pd.DataFrame(json.load(outfile))[0:100]

    print('Testing:')
    print((split_dataset1.fillna(0) != split_dataset2.fillna(0)).any().any())
