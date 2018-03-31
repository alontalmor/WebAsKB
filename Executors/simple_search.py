from config import *
from Executors.executor_base import ExecutorBase


class SimpleSearch(ExecutorBase):
    def __init__(self):
        ExecutorBase.__init__(self)
        self.version = "000"
        self.name = 'SimpQA'
        self.state_data = pd.DataFrame()
        self.number_of_waves = 1

    def gen_webanswer_first_wave(self,split_points):
        y1 = split_points[['question','answers']]
        y1.columns = ['question','answers']
        y1 = y1.to_dict(orient='rows')

        question_list = []
        for r1 in y1:
            if r1['question'] == r1['question']:
                #r1['goog_query'] = ' '.join([token for token in r1['question'].split(' ') \
                #                             if token.lower() not in set(stopwords.words('english'))])
                r1['goog_query'] = r1['question']
                question_list.append(r1)

        return question_list



    def proc_webanswer_wave_results(self,split_points,webanswer_dict):

        ex_split_points = split_points

        split_points[self.name + '_MRR'] = 0.0
        split_points[self.name + '_spans'] = None
        split_points[self.name + '_rc_conf'] = None
        split_points[self.name + '_spans'] = split_points[self.name + '_spans'].astype(object)
        split_points[self.name + '_rc_conf'] = split_points[self.name + '_rc_conf'].astype(object)

        for ind,question in ex_split_points.iterrows():
            curr_question = question['question']
            golden_answer_list = []
            for answer in question['answers']:
                golden_answer_list.append(answer['answer'])
                golden_answer_list += answer['aliases']

            if golden_answer_list[0] == None:
                continue

            answers = webanswer_dict[question['question']]

            if type(answers) == float:
                continue
            if len(answers)==0:
                continue
            answers = pd.DataFrame(answers)

            answers['spans'] = answers['spans'].str.lower().str.strip()

            #emitted_answers = answers[answers['scores'] > answers.iloc[0, 0] - 0.5]
            emitted_answers = answers

            matched_answers = self.qa_utils.compare_span_to_answer(list(emitted_answers['spans']), golden_answer_list,
                                                              curr_question)

            # computing new F1
            F1 = self.qa_utils.compute_MRR(matched_answers, golden_answer_list, emitted_answers)


            split_points.set_value(ind, self.name + '_MRR',F1)
            split_points.set_value(ind, self.name + '_spans',list(answers['spans']))
            split_points.set_value(ind, self.name + '_rc_conf',list(answers['scores']))

        return {'split_points':split_points,'question_list':[]}

    def proc_webanswer_second_wave_results(self, split_points, webanswer_dict):
        return {'split_points': split_points, 'question_list': []}