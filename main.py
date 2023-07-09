import argparse
import time
from run_network import Model
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', type=str, default='config.yaml', help='Constant configure')
parser.add_argument('-seed', type=int, default=50, help='Random seed')
parser.add_argument('-dm', type=int, default=256, help='Model dimension')
parser.add_argument('-nh', type=int, default=4, help='Number of headers')
parser.add_argument('-nle', type=int, default=2, help='Number of encoder layers')
parser.add_argument('-nld', type=int, default=2, help='Number of decoder layers')
parser.add_argument('-drop', type=float, default=0.1, help='Dropout rate')

parser.add_argument('-b', type=int, default=64)
parser.add_argument('-e', type=int, default=30)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-w', type=float, default=0)
parser.add_argument('-pth', type=str, default='trained_models/test_basic_single.pth')
parser.add_argument('-task', type=str, default=None)

parser.add_argument('-s', type=str, default='instance')
parser.add_argument('-e2', type=int, default=30)
parser.add_argument('-lr2', type=float, default=1e-4)
parser.add_argument('-w2', type=float, default=0)
parser.add_argument('-pth2', type=str, default='trained_models/re_trans_model.pth')
parser.add_argument('-task2', type=str, default=None)

parser.add_argument('-src', type=str, default='datasets/data/')
parser.add_argument('-result_folder', type=str, default='results/')
parser.add_argument('-act', type=str, nargs='+', default=['v'], help='Operation actions, t: train model; e: test_model') 
parser.add_argument('-task3', type=str, default=None)
parser.add_argument('-pth3', type=str, default='trained_models/re_trans_model.pth')

args = parser.parse_args()

if len(args.act) > 4 or len(args.act) < 1:
    raise Exception("The number of operations must be greater than 0 and less than 5 [0 - 4]")

for mode in args.act:
    assert (mode in ['t', 'rt', 'e', 'v'])

model = Model(args)

pre = None

time_start = time.time()
for mode in args.act:
    print(" [ " + mode + " ] ")
    if mode == 't':
        model.set_task(args.task)
        model.train_all()
        pre = mode

    elif mode == 'rt':
        model.set_task(args.task2)
        model.retrain_classifiers()
        pre = mode

    elif mode == 'e':
        if pre == 'e':
            continue
        elif pre == 't':
            model.set_task(args.task)
            model.independent_test()

        elif pre == 'rt':
            model.set_task(args.task2)
            model.independent_test(pth=args.pth2)

        elif pre is None:
            model.set_task(args.task3)
            model.independent_test(pth=args.pth3)

        else:
            raise NotImplementedError()
        pre = mode

    elif mode == 'v':
        model.visualization(62, 'ABP', "trained_models/TPpred_tw_60_1.pth")  # 50, 292, 381, 926



time_end = time.time()
t = time_end - time_start
print(f"Cost time: {t // 60} min {t % 60} sec")
