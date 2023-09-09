import argparse
import os

parser = argparse.ArgumentParser(description="WDENet")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()


noiseL = 'noise_' + str(int(opt.test_noiseL))
logs_n = 'WDENet_' + str(int(opt.test_noiseL)) + '.pth'


def function():
    print('function_1')

print(os.path.join(opt.logdir, noiseL, logs_n))

print('insert_1')
print('insert_1')
print('insert_1')

<<<<<<< Updated upstream
<<<<<<< Updated upstream

<<<<<<< HEAD

print('insert5')

<<<<<<< Updated upstream
=======
def function_3():
    print('function_3')
>>>>>>> 9469d8c70ed0629e8e185d666aac799a3085af04
=======

def function_2():
    print('function_2')
=======

>>>>>>> Stashed changes
=======

>>>>>>> Stashed changes

>>>>>>> Stashed changes
