from config import *
import re

class WebKB_Utils():
    def __init__(self,in_filename='',load_feature_nlp_tools=False):
        pass

    def compute_F1(self,matched_answers, golden_answer_list, answers):
        true_positive = len(matched_answers)
        num_emitted = len(answers)
        number_of_correct_answers = len(golden_answer_list)

        Percision = float(true_positive) / num_emitted * 100
        Recall = float(true_positive) / number_of_correct_answers * 100

        if Percision + Recall == 0:
            F1 = 0
        else:
            F1 = 2 * (Percision * Recall) / (Percision + Recall)

        return F1

    def compute_P1(self,matched_answers, golden_answer_list, answers):
        P1 = 0
        if len(matched_answers)>0:
            if answers.iloc[0]['spans'] in list(matched_answers['span']):
                P1 = 100

        return P1

    def compute_MRR(self,matched_answers, golden_answer_list, answers):
        MRR = 0
        if len(matched_answers)>0:
            for i in range(len(answers)):
                if answers.iloc[i]['spans'] in list(matched_answers['span']):
                    MRR = 1.0/float(i+1)
                    break

        return MRR

    def compare_span_to_answer(self,spans,answers,question,question_annotated=None):
        """ Compares one answers to spans, multiple matches are possible
        """
        if len(spans)==0:
            return []


        found_answers = pd.DataFrame(columns=['span','answer','span_index'])
        spans_series = pd.Series(spans)
        pre_proc_answers = []
        answers = [answer.lower().strip() for answer in answers]
        for answer in answers:
            proc_answer = unicodedata.normalize('NFKD', answer).encode('ascii', 'ignore').decode(encoding='UTF-8')

            # removing common endings such as "f.c."
            proc_answer = re.sub(r'\W',' ',proc_answer).lower().strip()
            # removing The, a, an from begining of answer as proposed by SQuAD dataset answer comparison
            if proc_answer.startswith('the '):
                proc_answer = proc_answer[4:]
            if proc_answer.startswith('a '):
                proc_answer = proc_answer[2:]
            if proc_answer.startswith('an '):
                proc_answer = proc_answer[3:]



            pre_proc_answers.append(proc_answer)



        question = question.lower().strip()

        # processing question:
        #question_annotated = pd.DataFrame(question_annotated)

        # exact match:
        for pre_proc_answer,answer in zip(pre_proc_answers,answers):

            if answer in spans:
                exact_match_ind = spans.index(answer)
                found_answers = found_answers.append({'span_index':exact_match_ind,'answer':answer,'span':answer},ignore_index=True)

            if pre_proc_answer in spans:
                exact_match_ind = spans.index(pre_proc_answer)
                found_answers = found_answers.append({'span_index': exact_match_ind, 'answer': answer, 'span': pre_proc_answer},ignore_index=True)

            # year should match year.
            if question.find('year')>-1:
                year_in_answer = re.search('([1-2][0-9]{3})', answer)
                if year_in_answer is not None:
                    year_in_answer = year_in_answer.group(0)

                year_spans = spans_series[spans_series == year_in_answer]
                if len(year_spans)>0:
                    found_answers = found_answers.append(
                        {'span_index': year_spans.index[0], 'answer': answer, 'span': year_in_answer}, ignore_index=True)


        return found_answers.drop_duplicates()

