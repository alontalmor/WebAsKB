from config import *
from SplitQA import SplitQA
from noisy_supervision import NoisySupervision
from webaskb_ptrnet import WebAsKB_PtrNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("operation", help='available operations: "gen_noisy_sup","run_ptrnet" ,"train_ptrnet", "splitqa"')
parser.add_argument("--eval_set", help='available eval sets: "dev","test"')
args = parser.parse_args()

if args.eval_set is not None:
    config.EVALUATION_SET = args.eval_set

if args.operation == 'gen_noisy_sup':
    noisy_sup = NoisySupervision()
    noisy_sup.gen_noisy_supervision()
elif args.operation == 'run_ptrnet':
    ptrnet = WebAsKB_PtrNet()
    ptrnet.load_data()
    ptrnet.init()
    ptrnet.eval()
elif args.operation == 'train_ptrnet':
    config.PERFORM_TRAINING = True
    config.LOAD_SAVED_MODEL = False
    config.max_evalset_size = 2000
    ptrnet = WebAsKB_PtrNet()
    ptrnet.load_data()
    ptrnet.init()
    ptrnet.train()
elif args.operation == 'splitqa':
    config.PERFORM_TRAINING = False
    splitqa = SplitQA()
    splitqa.run_executors()
    splitqa.compute_final_results()
else:
    print('option not found, available operations: "gen_noisy_sup","run_ptrnet" ,"train_ptrnet", "splitqa"')