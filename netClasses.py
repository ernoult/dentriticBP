from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from main import rho


class dentriticNet(nn.Module):
    def __init__(self, args):
        super(dentriticNet, self).__init__()
        
        self.T = args.T
        self.dt = args.dt
        self.size_tab = args.size_tab
        self.ns = len(args.size_tab) - 1
        self.batch_size = args.batch_size

        #****specific to dentritic****#
        self.lr_pp = args.lr_pp
        self.lr_ip = args.lr_ip
        self.lr_pi = args.lr_pi
        self.ga = args.ga
        self.gb = args.gb
        self.gd = args.gd
        self.glk = args.glk
        self.gsom = args.gsom
        self.noise = args.noise
        self.tau_neu = args.tau_neu
        self.tau_syn = args.tau_syn
        self.freeze_feedback = args.freeze_feedback
        #*****************************#
                    
        wpf = nn.ModuleList([])
        wpb = nn.ModuleList([])
        wpi = nn.ModuleList([])
        wip = nn.ModuleList([])
       
        #Build forward pyramidal weights
        for i in range(self.ns):
            wpf.append(nn.Linear(args.size_tab[i], args.size_tab[i + 1], bias = True))

        #Build backward pyramidal weights
        for i in range(self.ns - 1):
            wpb.append(nn.Linear(args.size_tab[i + 2], args.size_tab[i + 1], bias = False))

        #Build (forward) pyramidal to interneuron weights
        for i in range(self.ns - 1):
            wip.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i + 2], bias = False))

        #Build (backward) pyramidal to interneuron weights
        for i in range(self.ns - 1):
            wpi.append(nn.Linear(args.size_tab[i + 2], args.size_tab[i + 1], bias = False))        
                                     
        self.wpf = wpf
        self.wpb = wpb
        self.wip = wip
        self.wpi = wpi
        

    def stepper(self, data, s, i, nudge = False, track_va = False, **kwargs):

        dsdt = []
        #*****dynamics of the interneurons*****#
        didt = []
        #**************************************#

        #***track apical voltage***#
        if track_va:
            va_topdown = []
            va_cancellation = []
        #**************************#


        #Compute derivative of the somatic membrane potential
        for k in range(len(s)):

            #Compute basal voltage
            if k == 0 :#visible layer
                vb = self.wpf[k](rho(data))
            else:
                vb = self.wpf[k](rho(s[k - 1]))

            
            #a)for hidden neurons
            if k < len(s) - 1:

                #Compute and optionally store apical voltage
                va = self.wpi[k](rho(i[k + 1])) + self.wpb[k](rho(s[k + 1]))
                if track_va:
                    va_topdown.append(self.wpb[k](rho(s[k + 1])))
                    va_cancellation.append(self.wpi[k](rho(i[k + 1])))

                #Compute total derivative (Eq. 1)
                dsdt.append( -self.glk*s[k] + self.gb*(vb - s[k]) + self.ga*(va - s[k]) + self.noise*torch.randn_like(s[k]))

            #b) for output neurons
            else:
                va = torch.zeros_like(s[k])

                #Compute total derivative (Eq. 1) *with ga = 0*:
                dsdt.append( -self.glk*s[k] + self.gb*(vb - s[k]) + self.noise*torch.randn_like(s[k]))

                #Nudging
                if nudge and ('target' in kwargs):
                    dsdt[k] = dsdt[k] + self.gsom*(kwargs['target'] - s[k])

        
            del va, vb


        #Compute derivative of the interneuron membrane potential
        for k in range(len(i)):
            if i[k] is not None:
                #Compute basal interneuron voltage
                vi = self.wip[k - 1](rho(s[k - 1]))
                
                #Compute total derivative (Eq. 2)
                didt.append(-self.glk*i[k] + self.gd*(vi - i[k]) + self.gsom*(s[k] - i[k]) + self.noise*torch.randn_like(i[k]))
            else:
                didt.append(None)

        #Update the values of the neurons
        for k in range(len(s)):
            s[k] = s[k] + self.dt*dsdt[k]

        for k in range(len(i)):
            if i[k] is not None:
                i[k] = i[k] + self.dt*didt[k]

        if not track_va:
            return s, i

        else:
            return s, i, [va_topdown, va_cancellation]


    def forward(self, data, s, nudge = False, **kwargs):

        T = self.T

        if beta == 0:
            for t in range(T):                      
                s = self.stepper(data, s)
        else:
            for t in range(Kmax):                      
                s = self.stepper(data, s, target, beta)
        return s                                      
        
    def initHidden(self, **kwargs):
        s = []
        #****initialize interneurons****#
        i = [] 
        #*******************************#

        for k in range(self.ns):
            s.append(torch.zeros(self.batch_size, self.size_tab[k + 1]))

        for k in range(self.ns):            
            if k == 0:
                i.append(None) #visible layer has no feedback
            else:
                i.append(torch.zeros(self.batch_size, self.size_tab[k + 1]))

        if 'device' in kwargs:
            for k in range(len(s)):
                s[k] = s[k].to(kwargs['device'])
            for k in range(len(i)):
                if i[k] is not None:
                    i[k] = i[k].to(kwargs['device'])
       
        return s, i


              
    def computeGradients(self, data, s, i):
        gradwpf = []
        gradwpf_bias = []
        gradwpb = []
        gradwip = []
        gradwpi = []
        for k in range(self.ns):
            if k == 0:
                vb = self.wpf[k](rho(data))
                vbhat = self.gb/(self.gb + self.glk + self.ga)*vb
                gradwpf.append((1/self.batch_size)*(torch.mm(torch.transpose(rho(s[k]) - rho(vbhat), 0, 1),rho(data))))
            else:
                vb =  self.wpf[k](rho(s[k - 1]))
                vbhat = self.gb/(self.gb + self.glk + self.ga)*vb
                gradwpf.append((1/self.batch_size)*(torch.mm(torch.transpose(rho(s[k]) - rho(vbhat), 0, 1),rho(s[k - 1]))))

            gradwpf_bias.append((1/self.batch_size)*(rho(s[k]) - rho(vbhat)).sum(0))


        '''
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
             
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[i]), 0, 1), rho(s[i + 1])) - torch.mm(torch.transpose(rho(seq[i]), 0, 1), rho(seq[i + 1])))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(rho(s[i]) - rho(seq[i])).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[-1]) - rho(seq[-1]), 0, 1), rho(data)))
        gradw_bias.append((1/(beta*batch_size))*(rho(s[-1]) - rho(seq[-1])).sum(0))
               
        return  gradw, gradw_bias
        '''

  
    def updateWeights(self, beta, data, s, seq):
        
        '''		
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i]
        '''



