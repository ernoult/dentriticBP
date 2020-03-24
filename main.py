import argparse
import torch.optim as optim
import pickle
import datetime

from netClasses import *
from plotFunctions import *

parser = argparse.ArgumentParser(description='Reproducing "Dentritic cortical microcircuits approximate the backpropagation algorithm"')
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    metavar='N',
    help='input batch size for training (default: 1)')
parser.add_argument(
    '--samples',
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
    '--initw',
    type=int,
    default=1,
    metavar='N',
help='weights sampled from U[-a, a](default: a=1)')                                               
parser.add_argument(
    '--device-label',
    type=int,
    default=-1,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--freeze-feedback',
    action='store_true',
    default=False, 
help='freeze the dynamics of the feedback weights (default: False)')
parser.add_argument(
    '--bias',
    action='store_true',
    default=False, 
help='use biases (default: False)')
parser.add_argument(
    '--size_tab_teacher',
    nargs = '+',
    type=float,
    default=[30, 20, 10],
    metavar='STT',
    help='architecture of the teacher net (default: [30, 20, 10])')
parser.add_argument(
    '--k_tab',
    nargs = '+',
    type=float,
    default=[2, 10],
    metavar='STT',
    help='architecture of the teacher net (default: [30, 20, 10])')
parser.add_argument(
    '--action',
    type=str,
    default='fig1',
    metavar='ACT',
    help='activation function (default: logexp)')
parser.add_argument(
    '--init-selfpred',
    action='store_true',
    default=False, 
help='initialize weights properly for self-prediction (default: False)') 

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

    if args.action == 'fig1':

        #Build the net
        net = dentriticNet(args)
        net.to(device)
        net.train()

        with torch.no_grad():

            #Initialize neuron values
            s, i = net.initHidden(device = device)
                                  
            #**************LEARN THE SELF-PREDICTING REGIME**************#
            '''
            net.lr_ip = [0.0002375]
            net.lr_pi = [0.0005]	   
	
            for n in range(args.samples):
                print('Learning self-prediction, sample {}'.format(1 + n))

                #Pick a random sample
                data = torch.rand(args.batch_size, args.size_tab[0], device = device)

                if n == 0:
                    data_trace = data.clone()
                    data_trace_hist = data_trace.unsqueeze(2)

                for t in range(args.T):
                    #low-pass filter the data
                    data_trace +=  (args.dt/args.tau_neu)*(- data_trace + data)
                    data_trace_hist = torch.cat((data_trace_hist, data_trace.unsqueeze(2)), dim = 2)
                    
                    #Step the neural network
                    s, i, va = net.stepper(data_trace, s, i, track_va = True)

                    #Track apical potential, neurons and synapses
                    va_topdown, va_cancellation = va
                    if (t == 0) and (n == 0):
                        #Initialize the tabs with initial values
                        va_topdown_hist = net.initHist(va_topdown)
                        va_cancellation_hist = net.initHist(va_cancellation)
                        s_hist = net.initHist(s)
                        wpf_hist = net.initHist(net.wpf, param = True)
                        wpb_hist = net.initHist(net.wpb, param = True)
                        wpi_hist = net.initHist(net.wpi, param = True)
                        wip_hist= net.initHist(net.wip, param = True)
                                               
                    else:
                        #Update the tabs with the current values
                        va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
                        va_cancellation_hist = net.updateHist(va_cancellation_hist, va_cancellation)
                        s_hist = net.updateHist(s_hist, s)
                        wpf_hist = net.updateHist(wpf_hist, net.wpf, param = True)
                        wpb_hist = net.updateHist(wpb_hist, net.wpb, param = True)
                        wpi_hist = net.updateHist(wpi_hist, net.wpi, param = True)
                        wip_hist = net.updateHist(wip_hist, net.wip, param = True)
                    
                    #Update the pyramidal-to-interneuron weights (NOT the pyramidal-to-pyramidal weights !)
                    net.updateWeights(data, s, i, selfpredict = True)
            '''

            #**************LEARN INPUT-TARGET ASSOCIATION**************#
            
            net.lr_ip = [0.0011875]
            net.lr_pp = [0.0011875, 0.0005]            
            data = torch.rand(args.batch_size, args.size_tab[0], device = device)
            target = torch.rand(args.batch_size, args.size_tab[-1], device = device)

            data_trace = data.clone()
            data_trace_hist = data_trace.unsqueeze(2)

            print('Learning input-target association ...')
            for t in range(args.T):
                #print('t = {} ms'.format(0.1*t))
                #low-pass filter the data
                data_trace +=  (args.dt/args.tau_neu)*(- data_trace + data)
                data_trace_hist = torch.cat((data_trace_hist, data_trace.unsqueeze(2)), dim = 2)
                
                #Step the neural network
                s, i, va = net.stepper(data_trace, s, i, target = target, track_va = True)

                #Track apical potential, neurons and synapses
                va_topdown, va_cancellation = va
                if (t == 0):
                    #Initialize the tabs with initial values
                    va_topdown_hist = net.initHist(va_topdown)
                    va_cancellation_hist = net.initHist(va_cancellation)
                    s_hist = net.initHist(s)
                    wpf_hist = net.initHist(net.wpf, param = True)
                    wpb_hist = net.initHist(net.wpb, param = True)
                    wpi_hist = net.initHist(net.wpi, param = True)
                    wip_hist= net.initHist(net.wip, param = True)
                else:
                    va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
                    va_cancellation_hist = net.updateHist(va_cancellation_hist, va_cancellation)
                    s_hist = net.updateHist(s_hist, s)
                    wpf_hist = net.updateHist(wpf_hist, net.wpf, param = True)
                    wpb_hist = net.updateHist(wpb_hist, net.wpb, param = True)
                    wpi_hist = net.updateHist(wpi_hist, net.wpi, param = True)
                    wip_hist = net.updateHist(wip_hist, net.wip, param = True)

                #Update the pyramidal-to-interneuron weights (INCLUDING the pyramidal-to-pyramidal weights !)
                net.updateWeights(data, s, i, freeze_feedback = True)
                        			
                        
    
            #Plot the apical potential neuron-wise
            plot_results(args, data_trace = data_trace_hist, 
                        va_topdown = va_topdown_hist, 
                        va_cancellation = va_cancellation_hist)
            

            #Plot the neuron traces
            plot_results(args, data_trace = data_trace_hist,
                        s = s_hist)
                        
            #Plot the synapse traces
            plot_results(args, wpf = wpf_hist, wpb = wpb_hist,
                        wpi = wpi_hist, wip = wip_hist)

            plt.show()


    if args.action == 'fig2':

            #Build the net
            args.initw = 0.1
            net = dentriticNet(args)
            net.to(device)
            net.train()
            net.lr_ip = [0.0011875]
            net.lr_pi = [0.0059375]
            net.lr_pp = [0.0011875, 0.0005]
            net.noise = 0.3

            #Build the teacher net
            teacherNet = teacherNet(args)

            data = torch.rand(args.batch_size, args.size_tab[0], device = device)

            y = teacherNet.forward(data)

            with torch.no_grad():

                #Initialize neuron values
                s, i = net.initHidden(device = device)
                                      
                for n in range(args.samples):
                    print('Learning non-linear function, sample {}'.format(1 + n))
                    #Pick a random sample
                    data = torch.rand(args.batch_size, args.size_tab[0], device = device)

                    #***********WATCHOUT************#
                    target = teacherNet.forward(data)
                    #*******************************#

                    if n == 0:
                        data_trace = data.clone()
                        data_trace_hist = data_trace.unsqueeze(2)
                        target_hist = target.clone().unsqueeze(2)
                
                    for t in range(args.T):
                        #print('t = {} ms'.format(0.1*t))

                        #low-pass filter the data
                        data_trace +=  (args.dt/args.tau_neu)*(- data_trace + data)
                        data_trace_hist = torch.cat((data_trace_hist, data_trace.unsqueeze(2)), dim = 2)

                        #track the target output
                        target_hist = torch.cat((target_hist, target.unsqueeze(2)), dim = 2)
                        
                        #Step the neural network
                        s, i, va = net.stepper(data_trace, s, i, target = target, track_va = True)

                        #Track apical potential, neurons and synapses
                        va_topdown, va_cancellation = va

                        if (t == 0) and (n == 0):
                            #Initialize the tabs with initial values
                            va_topdown_hist = net.initHist(va_topdown)
                            va_cancellation_hist = net.initHist(va_cancellation)
                            s_hist = net.initHist(s)
                            wpf_hist = net.initHist(net.wpf, param = True)
                            wpb_hist = net.initHist(net.wpb, param = True)
                            wpi_hist = net.initHist(net.wpi, param = True)
                            wip_hist= net.initHist(net.wip, param = True)

                        else:                                              
                            va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
                            va_cancellation_hist = net.updateHist(va_cancellation_hist, va_cancellation)
                            s_hist = net.updateHist(s_hist, s)
                            wpf_hist = net.updateHist(wpf_hist, net.wpf, param = True)
                            wpb_hist = net.updateHist(wpb_hist, net.wpb, param = True)
                            wpi_hist = net.updateHist(wpi_hist, net.wpi, param = True)
                            wip_hist = net.updateHist(wip_hist, net.wip, param = True)

                        #Update the pyramidal-to-interneuron weights (INCLUDING the pyramidal-to-pyramidal weights !)
                        net.updateWeights(data, s, i, freeze_feedback = True)
                                			
                                
            
                #Plot the apical potential neuron-wise
                plot_results(args, data_trace = data_trace_hist, 
                            va_topdown = va_topdown_hist, 
                            va_cancellation = va_cancellation_hist,
                            s = s_hist,
                            target = target_hist)
                

                #Plot the neuron traces
                plot_results(args, data_trace = data_trace_hist,
                            s = s_hist)
                            
                #Plot the synapse traces
                plot_results(args, wpf = wpf_hist, wpb = wpb_hist,
                            wpi = wpi_hist, wip = wip_hist)

                plt.show()

