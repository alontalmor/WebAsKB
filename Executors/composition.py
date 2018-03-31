from config import *
from Executors.executor_base import ExecutorBase


class Composition(ExecutorBase):
    def __init__(self):
        ExecutorBase.__init__(self)
        self.version = "000"
        self.name = 'SplitQA_composition'
        self.state_data = pd.DataFrame()
        self.number_of_waves = 1

    def gen_webanswer_first_wave(self,split_points):
        question_list = []

        y1 = split_points[split_points['comp'] == 'composition'][['split_part1']]
        y1.columns = ['question']
        question_list += y1.to_dict(orient='rows')

        for question in question_list:
            # separating between the question and goog_query
            question['goog_query'] = question['question']

        return question_list

    def proc_webanswer_wave_results(self,split_points,webanswer_dict):

        ex_split_points = split_points[split_points['comp'] == 'composition']

        split_points[self.name + '_MRR'] = 0.0
        split_points[self.name + '_spans'] = None
        split_points[self.name + '_rc_conf'] = None
        split_points[self.name + '_spans'] = split_points[self.name + '_spans'].astype(object)
        split_points[self.name + '_rc_conf'] = split_points[self.name + '_rc_conf'].astype(object)

        question_list = []
        for ind,question in ex_split_points.iterrows():

            if question['split_part1']!=question['split_part1']:
                continue
            answers1 = webanswer_dict[question['split_part1']]

            if len(answers1) == 0:
                continue

            chosen_answer = ''
            if len(answers1) > 0:
                chosen_answer = answers1.iloc[0, 1]

            second_q = question['split_part2'].replace('%composition',chosen_answer)

            goog_query = second_q

            question_list.append({'question':second_q,'goog_query':goog_query,'answers':question['answers']})

        return {'split_points':split_points,'question_list':question_list}


    def proc_webanswer_second_wave_results(self,split_points,webanswer_dict):
        ex_split_points = split_points[split_points['comp'] == 'composition']

        for ind,question in ex_split_points.iterrows():
            curr_question = question['question']
            golden_answer_list = []
            for answer in question['answers']:
                golden_answer_list.append(answer['answer'])
                golden_answer_list += answer['aliases']

            if golden_answer_list[0] == None:
                continue

            if question['split_part1']!=question['split_part1']:
                continue
            answers1 = pd.DataFrame(webanswer_dict[question['split_part1']])

            if len(answers1) == 0:
                continue

            chosen_answer = ''
            if len(answers1) > 0:
                chosen_answer = answers1.iloc[0, 1]

            second_q = question['split_part2'].replace('%composition', chosen_answer)

            answers2 = pd.DataFrame(webanswer_dict[second_q])


            if len(answers2) > 0:
                answers2['spans'] = answers2['spans'].str.lower().str.strip()
                emitted_answers = answers2

                matched_answers = self.qa_utils.compare_span_to_answer(list(emitted_answers['spans']), golden_answer_list,
                                                                  curr_question)

                # computing new F1
                F1 = self.qa_utils.compute_MRR(matched_answers, golden_answer_list, emitted_answers)
            else:
                F1 = 0


            split_points.set_value(ind, self.name + '_MRR', F1)
            if len(answers2)>0:
                split_points.set_value(ind, self.name + '_spans', list(answers2['spans']))
                split_points.set_value(ind, self.name + '_rc_conf', list(answers2['scores']))

        return {'split_points':split_points,'question_list':[]}


