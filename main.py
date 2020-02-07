import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime

from netClasses import *

parser = argparse.ArgumentParser(description='Reproducing "Dentritic cortical microcircuits approximate the backpropagation algorithm"')
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    metavar='N',
    help='input batch size for training (default: 1)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
help='number of epochs to train (default: 1)')      
parser.add_argument(
    '--dt',
    type=float,
    default=0.1,
    metavar='DT',
    help='time discretization (default: 0.1)') 
parser.add_argument(
    '--T',
    type=int,
    default=1000,
    metavar='T',
    help='number of time steps per sample (default: 1000)')
parser.add_argument(
    '--size_tab',
    nargs = '+',
    type=int,
    default=[30, 20, 10],
    metavar='ST',
    help='network topology (default: [30, 20, 10])')
parser.add_argument(
    '--lr_pp',
    nargs = '+',
    type=float,
    default=[0],
    metavar='LR',
    help='learning rates for the P to P synapses (default: [0])')
parser.add_argument(
    '--lr_ip',
    nargs = '+',
    type=float,
    default=[0.0002375],
    metavar='LR',
    help='learning rates for the P to P synapses (default: [0.0002375])')
parser.add_argument(
    '--lr_pi',
    nargs = '+',
    type=float,
    default=[0.0005],
    metavar='LR',
    help='learning rates for the P to P synapses (default: [0.0059375])')
parser.add_argument(
    '--ga',
    type=float,
    default=0.8,
    metavar='G',
    help='apical conductance (default: 0.8)')
parser.add_argument(
    '--gb',
    type=float,
    default=1,
    metavar='G',
    help='basal conductance (default: 1)')
parser.add_argument(
    '--gd',
    type=float,
    default=1,
    metavar='G',
    help='basal conductance (default: 1)')
parser.add_argument(
    '--glk',
    type=float,
    default=0.1,
    metavar='G',
    help='leakage conductance (default: 0.1)')
parser.add_argument(
    '--gsom',
    type=float,
    default=0.1,
    metavar='G',
    help='somatic conductance (default: 0.1)')
parser.add_argument(
    '--noise',
    type=float,
    default=0.1,
    metavar='N',
    help='noise level (default: 0.1)')
parser.add_argument(
    '--tau_neu',
    type=float,
    default=3,
    metavar='N',
    help='time constant of the input patterns (default: 3)')
parser.add_argument(
    '--tau_syn',
    type=float,
    default=30,
    metavar='N',
    help='time constant of the synapses (default: 30)')
parser.add_argument(
    '--activation-function',
    type=str,
    default='logexp',
    metavar='ACTFUN',
    help='activation function (default: logexp)')                                               
parser.add_argument(
    '--device-label',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--freeze-feedback',
    action='store_true',
    default=False, 
help='freeze the dynamics of the feedback weights (default: False)')

args = parser.parse_args()


if  args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

elif args.activation_function == 'hardsigm':
    def rho(x):
        return x.clamp(min = 0).clamp(max = 1)

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)

elif args.activation_function == 'logexp':
    def rho(x):
        return torch.log(1 + torch.exp(x))
            
                    
if __name__ == '__main__':
  
    #Define the device
    if args.device_label >= 0:    
        device = torch.device("cuda:"+str(args.device_label)+")")
    else:
        device = torch.device("cpu")

    #Build the net
    net = dentriticNet(args)
    net.to(device)
    net.train()

    #Check net parameters dimensions
    '''
    print(net)
    '''    

    #Generate random input on the visible layer
    data = torch.rand(args.batch_size, args.size_tab[0], device = device)

    #Initialize hidden units
    s, i = net.initHidden(device = device)

    
    #Check hidden units dimensions
    '''
    for ind, s_temp in enumerate(s):
        print('Layer {}: dimension {}'.format(ind, s_temp.size()))
    for ind, i_temp in enumerate(i):
        if i_temp is None:
            print('Layer {}: None'.format(ind))
        else:
            print('Layer {}: dimension {}'.format(ind, i_temp.size()))
    '''       

    #Test stepper
    '''
    s, i = net.stepper(data, s, i)
    '''

    #Test weight update
    with torch.no_grad():
        net.updateWeights(data, s, i)



    print('Everything is all right until here !')
    






