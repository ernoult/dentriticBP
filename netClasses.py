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
            if args.bias:
                wpf.append(nn.Linear(args.size_tab[i], args.size_tab[i + 1]))
            else:
                wpf.append(nn.Linear(args.size_tab[i], args.size_tab[i + 1], bias = False))

            torch.nn.init.uniform_(wpf[i].weight, a = -args.initw, b = args.initw)

        #Build backward pyramidal weights
        for i in range(self.ns - 1):
            wpb.append(nn.Linear(args.size_tab[i + 2], args.size_tab[i + 1], bias = False))
            torch.nn.init.uniform_(wpb[i].weight, a = -args.initw, b = args.initw)

        #Build (forward) pyramidal to interneuron weights
        for i in range(self.ns - 1):
            wip.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i + 2], bias = False))
            torch.nn.init.uniform_(wip[i].weight, a = -args.initw, b = args.initw)

        #Build (backward) pyramidal to interneuron weights
        for i in range(self.ns - 1):
            wpi.append(nn.Linear(args.size_tab[i + 2], args.size_tab[i + 1], bias = False))
            torch.nn.init.uniform_(wpi[i].weight, a = -args.initw, b = args.initw)      
                                     
        self.wpf = wpf
        self.wpb = wpb
        self.wip = wip
        self.wpi = wpi
        

    def initHist(self, tab, param = False):
        hist = []
        if not param:
            for k in range(len(tab)):
                hist_temp = tab[k].unsqueeze(2)
                hist.append(hist_temp)
                del hist_temp
        else:
            for k in range(len(tab)):
                hist_temp = tab[k].weight.unsqueeze(2)
                hist.append(hist_temp)
                del hist_temp            

        return hist

    def updateHist(self, hist, tab, param = False):
        if not param:
            for k in range(len(tab)):                           
                hist[k] = torch.cat((hist[k], tab[k].unsqueeze(2)), dim = 2)
        else:
            for k in range(len(tab)):                           
                hist[k] = torch.cat((hist[k], tab[k].weight.unsqueeze(2)), dim = 2)            

        return hist    

    def stepper(self, data, s, i, track_va = False, **kwargs):

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

                del va

            #b) for output neurons
            else:

                #Compute total derivative (Eq. 1) *with ga = 0*:
                dsdt.append( -self.glk*s[k] + self.gb*(vb - s[k]) + self.noise*torch.randn_like(s[k]))

                #Nudging
                if 'target' in kwargs:
                    dsdt[k] = dsdt[k] + self.gsom*(kwargs['target'] - s[k])

        
            del vb


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
                gradwpf.append((1/self.batch_size)*(torch.mm(torch.transpose(rho(s[k]) - rho(vbhat), 0, 1), rho(s[k - 1]))))

            gradwpf_bias.append((1/self.batch_size)*(rho(s[k]) - rho(vbhat)).sum(0))

	    del vb, vbhat

	    for k in range(self.ns - 1):
                vi = self.wip[k](rho(s[k]))
                vihat = self.gd/(self.gd + self.glk)*vi
	        gradwip.append((1/self.batch_size)*(torch.mm(torch.transpose(rho(i[k + 1]) - rho(vihat), 0, 1), rho(s[k]))))
                
	        va = self.wpi[k](rho(i[k + 1])) + self.wpb[k](rho(s[k + 1]))
                gradwpi.append((1/self.batch_size)*(torch.mm(torch.transpose(-va, 0, 1), rho(i[k + 1]))))

	        vtdhat = self.wpb[k](rho(s[k + 1]))
                gradwpb.append((1/self.batch_size)*(torch.mm(torch.transpose(rho(s[k]) - rho(vtdhat), 0, 1), rho(s[k + 1]))))
       
                del vi, vihat, va, vtdhat

        return gradwpf, gradwpf_bias, gradwpb, gradwpi, gradwip
             

    def updateWeights(self, data, s, i, selfpredict = False, freeze_feedback = False):

        gradwpf, gradwpf_bias, gradwpb, gradwpi, gradwip = self.computeGradients(data, s, i)

        if not selfpredict:
            for k in range(len(self.wpf)):
                self.wpf[k].weight += self.lr_pp[k]*self.dt*gradwpf[k] - (self.dt/self.tau_syn)*self.wpf[k].weight
                if self.wpf[k].bias is not None:
                    self.wpf[k].bias += self.lr_pp[k]*self.dt*gradwpf_bias[k] - (self.dt/self.tau_syn)*self.wpf[k].bias
                if not freeze_feedback:
                    for k in range(len(self.wpb)):
                        self.wpb[k].weight += self.lr_pp[k]*self.dt*gradwpb[k] - (self.dt/self.tau_syn)*self.wpb[k].weight
            
        for k in range(len(self.wpb)):
            self.wpi[k].weight += self.lr_pi[k]*self.dt*gradwpi[k] - (self.dt/self.tau_syn)*self.wpi[k].weight
            self.wip[k].weight += self.lr_ip[k]*self.dt*gradwip[k] - (self.dt/self.tau_syn)*self.wip[k].weight


class teacherNet(nn.Module):
    def __init__(self, args):
        super(teacherNet, self).__init__()
        
        self.size_tab = args.size_tab_teacher

        self.gamma = 0.1
        self.beta = 1
        self.theta = 3
        self.k_tab = args.k_tab
                    
        w = nn.ModuleList([])

        #Build weights
        for i in range(len(args.size_tab_teacher) - 1):
            w.append(nn.Linear(args.size_tab_teacher[i], args.size_tab_teacher[i + 1], bias = False))
            torch.nn.init.uniform_(w[i].weight, a = -1, b = 1)
               
        self.w = w

    def rho(self, x):
        return self.gamma*torch.log(1 + torch.exp(self.beta*(x - self.theta)))

    def forward(self, x):
        a = x
        for i in range(len(self.w)):
            a = self.rho(self.k_tab[i]*self.w[i](a))

        return a

       

