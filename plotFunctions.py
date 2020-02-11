import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch



def plot_results(args, **kwargs):
    
    if 'va_topdown' in kwargs:
        data_trace_hist = kwargs['data_trace']
        va_topdown_hist = kwargs['va_topdown']
        va_cancellation_hist = kwargs['va_cancellation']

        if not ('target' in kwargs):
            fig = plt.figure(figsize = (15, 2*(len(args.size_tab) - 1)))
            plt.rcParams.update({'font.size': 12})
            N_neu = 5
            #Loop over layers
            for i in range(len(args.size_tab) - 1):
                #Loop over randomly picked neurons of each layer
                for j in range(N_neu + 1):                    
                    #Visible layer
                    if (i == 0):
                        if (j < N_neu):
                            plt.subplot(len(args.size_tab) - 1, N_neu + 1, (N_neu + 1)*i + j + 1)
                            ind_neu = np.random.randint(data_trace_hist.size(1))
                            plt.plot(0.1*np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)), data_trace_hist[0, ind_neu, :].cpu().numpy(), 
                                    color = 'green', linewidth = 3, alpha = 0.5)
                            if j == 0:
                                plt.ylabel('Sensory input')
                            plt.xlabel('Time (ms)')
                            plt.grid()
                    
                    #Hidden layer
                    else:
                        plt.subplot(len(args.size_tab) - 1, N_neu + 1, (N_neu + 1)*i + j + 1)
                        if j < N_neu:
                            ind_neu = np.random.randint(va_topdown_hist[i - 1].size(1))

                            plt.plot(0.1*np.linspace(0, va_topdown_hist[i - 1].size(2) - 1, va_topdown_hist[i - 1].size(2)), va_topdown_hist[i - 1][0, ind_neu, :].cpu().numpy(), 
                                    color = 'blue', linewidth = 3, alpha = 0.5, label = 'Topdown feedback')
                            plt.plot(0.1*np.linspace(0, va_cancellation_hist[i - 1].size(2) - 1, va_cancellation_hist[i - 1].size(2)), 
                                    va_cancellation_hist[i - 1][0, ind_neu, :].cpu().numpy(), 
                                    color = 'red', linewidth = 3, alpha = 0.5, label = 'Lateral cancellation')
                            plt.plot(0.1*np.linspace(0, va_cancellation_hist[i - 1].size(2) - 1, va_cancellation_hist[i - 1].size(2)), 
                                    (va_cancellation_hist[i - 1][0, ind_neu, :] + va_topdown_hist[i - 1][0, ind_neu, :]).cpu().numpy(), 
                                    color = 'grey', linewidth = 3, alpha = 0.5, label = 'Apical potential')
                            plt.xlabel('Time (ms)')
                            if j == 0:
                                plt.ylabel('Layer {}'.format(i))
                            plt.title('Neuron {}'.format(1 + j))
                            plt.grid()
                        else:
                            plt.plot(0.1*np.linspace(0, va_topdown_hist[i - 1].size(2) - 1, va_topdown_hist[i - 1].size(2)), 
                                    ((va_cancellation_hist[i - 1] + va_topdown_hist[i - 1])**2).sum(1).squeeze(0),
                                     color = 'grey', linewidth = 3, alpha = 0.5, label = 'Apical potential throughout time')
                            plt.xlabel('Time (ms)')
                            plt.title(' Apical pot of layer ' + str(i))
                            plt.grid()
                
            fig.tight_layout()

        else:
            s_hist = kwargs['s']
            target_hist = kwargs['target']

            fig = plt.figure(figsize = (15, 2*(len(args.size_tab) - 1) + 2))
            plt.rcParams.update({'font.size': 12})
            N_neu = 5
            #Loop over layers
            for i in range(len(args.size_tab) - 1):
                #Loop over randomly picked neurons of each layer
                for j in range(N_neu + 1):                    
                    #Visible layer
                    if (i == 0):
                        if (j < N_neu):
                            plt.subplot(len(args.size_tab), N_neu + 1, (N_neu + 1)*i + j + 1)
                            ind_neu = np.random.randint(data_trace_hist.size(1))
                            plt.plot(0.1*np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)), data_trace_hist[0, ind_neu, :].cpu().numpy(), 
                                    color = 'green', linewidth = 3, alpha = 0.5)
                            if j == 0:
                                plt.ylabel('Sensory input')
                            plt.xlabel('Time (ms)')
                            plt.grid()
                    
                    #Hidden layer
                    else:
                        plt.subplot(len(args.size_tab), N_neu + 1, (N_neu + 1)*i + j + 1)
                        if j < N_neu:
                            ind_neu = np.random.randint(va_topdown_hist[i - 1].size(1))

                            plt.plot(0.1*np.linspace(0, va_topdown_hist[i - 1].size(2) - 1, va_topdown_hist[i - 1].size(2)), va_topdown_hist[i - 1][0, ind_neu, :].cpu().numpy(), 
                                    color = 'blue', linewidth = 3, alpha = 0.5, label = 'Topdown feedback')
                            plt.plot(0.1*np.linspace(0, va_cancellation_hist[i - 1].size(2) - 1, va_cancellation_hist[i - 1].size(2)), 
                                    va_cancellation_hist[i - 1][0, ind_neu, :].cpu().numpy(), 
                                    color = 'red', linewidth = 3, alpha = 0.5, label = 'Lateral cancellation')
                            plt.plot(0.1*np.linspace(0, va_cancellation_hist[i - 1].size(2) - 1, va_cancellation_hist[i - 1].size(2)), 
                                    (va_cancellation_hist[i - 1][0, ind_neu, :] + va_topdown_hist[i - 1][0, ind_neu, :]).cpu().numpy(), 
                                    color = 'grey', linewidth = 3, alpha = 0.5, label = 'Apical potential')
                            plt.xlabel('Time (ms)')
                            if j == 0:
                                plt.ylabel('Layer {}'.format(i))
                            plt.title('Neuron {}'.format(1 + j))
                            plt.grid()
                        else:
                            plt.plot(0.1*np.linspace(0, va_topdown_hist[i - 1].size(2) - 1, va_topdown_hist[i - 1].size(2)), 
                                    ((va_cancellation_hist[i - 1] + va_topdown_hist[i - 1])**2).sum(1).squeeze(0),
                                     color = 'grey', linewidth = 3, alpha = 0.5, label = 'Apical potential throughout time')
                            plt.xlabel('Time (ms)')
                            plt.title(' Apical pot of layer ' + str(i))
                            plt.grid()

            ind_subplot = (N_neu + 1)*i + j + 2
            del i, j

            for j in range(N_neu):
                plt.subplot(len(args.size_tab), N_neu + 1, ind_subplot + j)
                
                plt.plot(0.1*np.linspace(0, s_hist[- 1].size(2) - 1, s_hist[- 1].size(2)), s_hist[- 1][0, j, :].cpu().numpy(), 
                        color = 'blue', linewidth = 3, alpha = 0.5)
                plt.plot(0.1*np.linspace(0, target_hist.size(2) - 1, target_hist.size(2)), target_hist[0, j, :].cpu().numpy(), 
                        color = 'blue', linewidth = 3, alpha = 0.8, linestyle = '--')
                plt.grid()
                plt.xlabel('Time (ms)')
                plt.title('Neuron {}'.format(1 + j))
                if j == 0:
                    plt.ylabel('Output layer')
                
            fig.tight_layout()

    elif  's' in kwargs:
        data_trace_hist = kwargs['data_trace']
        s_hist = kwargs['s']

        fig = plt.figure(figsize = (5, 2*(len(args.size_tab))))
        plt.rcParams.update({'font.size': 12})

        #Loop over layers
        for i in range(len(args.size_tab)):
            plt.subplot(len(args.size_tab), 1, 1 + i)
          
            #Visible layer
            if (i == 0):                    
                plt.plot(0.1*np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)), data_trace_hist.mean(1).squeeze(0).cpu().numpy(), 
                        color = 'green', linewidth = 3, alpha = 0.8)

                for j in range(10):
                    plt.plot(0.1*np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)), data_trace_hist[0, j, :].cpu().numpy(), 
                            color = 'green', linewidth = 1.5, alpha = 0.2)                        

                plt.ylabel('Sensory input')
                plt.xlabel('Time (ms)')
                plt.title('Neuron trajectories')
                plt.grid()
                
            #Hidden layer
            else:
                plt.plot(0.1*np.linspace(0, s_hist[i - 1].size(2) - 1, s_hist[i - 1].size(2)), s_hist[i - 1].mean(1).squeeze(0).cpu().numpy(), 
                        color = 'blue', linewidth = 3, alpha = 0.8)

                for j in range(10):
                    plt.plot(0.1*np.linspace(0, s_hist[i - 1].size(2) - 1, s_hist[i - 1].size(2)), s_hist[i - 1][0, j, :].cpu().numpy(), 
                            color = 'blue', linewidth = 1.5, alpha = 0.2)

                plt.xlabel('Time (ms)')
                plt.ylabel('Layer {}'.format(i))
                plt.grid()

        fig.tight_layout()

    elif 'wpf' in kwargs:
        wpf_hist = kwargs['wpf']
        wpb_hist = kwargs['wpb']
        wpi_hist = kwargs['wpi']
        wip_hist = kwargs['wip']
        fig = plt.figure(figsize = (12, (len(args.size_tab))))
        plt.rcParams.update({'font.size': 12})

        #Loop over layers
        for i in range(len(args.size_tab) - 1):

            #wpf
            plt.subplot(len(args.size_tab) - 1, 4, 1 + 4*i)

            plt.plot(0.1*np.linspace(0, wpf_hist[i].size(2) - 1, wpf_hist[i].size(2)), wpf_hist[i].mean(0).mean(0).cpu().numpy(), 
                    color = 'red', linewidth = 3, alpha = 0.8)

            for j in range(10):
                ind_0, ind_1 = np.random.randint(wpf_hist[i].size(0)), np.random.randint(wpf_hist[i].size(1))
                plt.plot(0.1*np.linspace(0, wpf_hist[i].size(2) - 1, wpf_hist[i].size(2)), wpf_hist[i][ind_0, ind_1, :].cpu().numpy(), 
                        color = 'red', linewidth = 1.5, alpha = 0.2)                        

            if i == len(args.size_tab) - 2:
                plt.xlabel(r'$W^{\rm pp, forward}$')
            else:
                plt.xlabel('Time (ms)')
            
            plt.ylabel('Layer {}'.format(1 + i))

            plt.grid()
                
            #Hidden layer
            if i > 0:
                #wpb
                plt.subplot(len(args.size_tab) - 1, 4, 2 + 4*i)                 
                plt.plot(0.1*np.linspace(0, wpb_hist[i - 1].size(2) - 1, wpb_hist[i - 1].size(2)), wpb_hist[i - 1].mean(0).mean(0).cpu().numpy(), 
                        color = 'blue', linewidth = 3, alpha = 0.8)

                for j in range(10):
                    ind_0, ind_1 = np.random.randint(wpb_hist[i - 1].size(0)), np.random.randint(wpb_hist[i - 1].size(1))
                    plt.plot(0.1*np.linspace(0, wpb_hist[i - 1].size(2) - 1, wpb_hist[i - 1].size(2)), wpb_hist[i - 1][ind_0, ind_1, :].cpu().numpy(), 
                            color = 'blue', linewidth = 1.5, alpha = 0.2)                        

                if i == len(args.size_tab) - 2:
                    plt.xlabel(r'$W^{\rm pp, backward}$')
                else:
                    plt.xlabel('Time (ms)')

                plt.grid()

                #wpi
                plt.subplot(len(args.size_tab) - 1, 4, 3 + 4*i)                 
                plt.plot(0.1*np.linspace(0, wpi_hist[i - 1].size(2) - 1, wpi_hist[i - 1].size(2)), wpi_hist[i - 1].mean(0).mean(0).cpu().numpy(), 
                        color = 'orange', linewidth = 3, alpha = 0.8)

                for j in range(10):
                    ind_0, ind_1 = np.random.randint(wpi_hist[i - 1].size(0)), np.random.randint(wpi_hist[i - 1].size(1))
                    plt.plot(0.1*np.linspace(0, wpi_hist[i - 1].size(2) - 1, wpi_hist[i - 1].size(2)), wpi_hist[i - 1][ind_0, ind_1, :].cpu().numpy(), 
                            color = 'orange', linewidth = 1.5, alpha = 0.2)                        

                
                if i == len(args.size_tab) - 2:
                    plt.xlabel(r'$W^{\rm pi}$')
                else:
                    plt.xlabel('Time (ms)')

                plt.grid()

                #wip
                plt.subplot(len(args.size_tab) - 1, 4, 4 + 4*i)                 
                plt.plot(0.1*np.linspace(0, wip_hist[i - 1].size(2) - 1, wip_hist[i - 1].size(2)), wip_hist[i - 1].mean(0).mean(0).cpu().numpy(), 
                        color = 'green', linewidth = 3, alpha = 0.8)

                for j in range(10):
                    ind_0, ind_1 = np.random.randint(wip_hist[i - 1].size(0)), np.random.randint(wip_hist[i - 1].size(1))
                    plt.plot(0.1*np.linspace(0, wip_hist[i - 1].size(2) - 1, wip_hist[i - 1].size(2)), wip_hist[i - 1][ind_0, ind_1, :].cpu().numpy(), 
                            color = 'green', linewidth = 1.5, alpha = 0.2)                        

                if i == len(args.size_tab) - 2:
                    plt.xlabel(r'$W^{\rm ip}$')
                else:
                    plt.xlabel('Time (ms)')

                plt.grid()		



        fig.tight_layout()    
        plt.show()    

              
