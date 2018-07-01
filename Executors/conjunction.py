from config import *
from Executors.executor_base import ExecutorBase


class Conjunction(ExecutorBase):
    def __init__(self):
        ExecutorBase.__init__(self)
        self.version = "000"
        self.name = 'SplitQA_conjunction'
        self.state_data = pd.DataFrame()
        self.number_of_waves = 1

    def gen_webanswer_first_wave(self,split_points):
        ex_split_points = split_points[split_points['comp'] == 'conjunction']
        y1 = ex_split_points[['split_part1','answers']]
        y1.columns = ['question','answers']
        y1 = y1.to_dict(orient='rows')
        y2 = ex_split_points[['split_part2','answers']]
        y2.columns = ['question','answers']
        y2 = y2.to_dict(orient='rows')

        question_list = []
        for r1, r2 in zip(y1, y2):
            if r1['question'] == r1['question'] and r2['question'] == r2['question']:

                # separating between the question and goog_query
                r1['goog_query'] = r1['question']
                r2['goog_query'] = r2['question']
                #r1['goog_query'] = ' '.join([token for token in r1['question'].split(' ') \
                #              if token.lower() not in set(stopwords.words('english'))])
                #r2['goog_query'] = ' '.join([token for token in r2['question'].split(' ') \
                #                             if token.lower() not in set(stopwords.words('english'))])

                question_list.append(r1)
                question_list.append(r2)

        return question_list

    def proc_webanswer_wave_results(self,split_points,webanswer_dict):

        ex_split_points = split_points[split_points['comp'] == 'conjunction']

        split_points[self.name + '_MRR'] = 0.0
        split_points[self.name + '_MRR_1'] = 0.0
        split_points[self.name + '_MRR_2'] = 0.0
        split_points[self.name + '_spans'] = None
        split_points[self.name + '_rc_conf'] = None
        split_points[self.name + '_spans'] = split_points[self.name + '_spans'].astype(object)
        split_points[self.name + '_rc_conf'] = split_points[self.name + '_rc_conf'].astype(object)

        for ind,question in ex_split_points.iterrows():
            if question['split_part1']!=question['split_part1'] or question['split_part2']!=question['split_part2']:
                continue

            curr_question = question['question']
            golden_answer_list = []
            for answer in question['answers']:
                golden_answer_list.append(answer['answer'])
                golden_answer_list += answer['aliases']

            if golden_answer_list[0] == None:
                continue

            answers1 = pd.DataFrame(webanswer_dict[question['split_part1']])
            answers2 = pd.DataFrame(webanswer_dict[question['split_part2']])

            if len(answers1) == 0 or len(answers2) == 0:
                continue

            answers1['spans'] = answers1['spans'].str.lower().str.strip()
            answers2['spans'] = answers2['spans'].str.lower().str.strip()

            joined_answers = answers1.merge(answers2, on='spans', how='inner')
            joined_answers['scores'] = joined_answers[['scores_x', 'scores_y']].max(axis=1)
            joined_answers = joined_answers.sort_values(by='scores', ascending=False)
            joined_answers = joined_answers[['scores', 'spans', 'scores_x', 'scores_y']]

            if len(joined_answers) > 0:
                #emitted_answers = joined_answers[joined_answers['scores'] > joined_answers.iloc[0, 0] - 0.5]
                emitted_answers = joined_answers

                matched_answers = self.qa_utils.compare_span_to_answer(list(emitted_answers['spans']), golden_answer_list,
                                                                  curr_question)

                # computing new F1
                F1 = self.qa_utils.compute_MRR(matched_answers, golden_answer_list, emitted_answers)

                emitted_answers1 = answers1[answers1['scores'] > answers1.iloc[0, 0] - 0.5]
                matched_answers = self.qa_utils.compare_span_to_answer(list(emitted_answers1['spans']),
                                                                       golden_answer_list,
                                                                       curr_question)
                F1_1 = self.qa_utils.compute_MRR(matched_answers, golden_answer_list, emitted_answers1)

                emitted_answers2 = answers2[answers2['scores'] > answers2.iloc[0, 0] - 0.5]
                matched_answers = self.qa_utils.compare_span_to_answer(list(emitted_answers2['spans']),
                                                                       golden_answer_list,
                                                                       curr_question)
                F1_2 = self.qa_utils.compute_MRR(matched_answers, golden_answer_list, emitted_answers2)
            else:
                F1 = 0
                F1_1 = 0
                F1_2 = 0

            split_points.set_value(ind, self.name + '_MRR',F1)
            split_points.set_value(ind, self.name + '_MRR_1',F1_1)
            split_points.set_value(ind, self.name + '_MRR_2',F1_2)
            if len(joined_answers) > 0:
                split_points.set_value(ind, self.name + '_spans',list(joined_answers['spans']))
                split_points.set_value(ind, self.name + '_rc_conf',list(joined_answers['scores']))

        return {'split_points':split_points,'question_list':[]}

    def proc_webanswer_second_wave_results(self, split_points, webanswer_dict):
        return {'split_points': split_points, 'question_list': []}















