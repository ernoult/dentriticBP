python main.py --action figs1 --lr_ip 0.0002375 --lr_pi 0.0005 --noise 0.1

python main.py --action fig1 --lr_ip 0.0011875 --lr_pi 0 --lr_pp 0.0005 0.0011875 --noise 0.1

python main.py --action fig2 --lr_ip 0.0011875 --lr_pi 0.0059375 --lr_pp 0.0005 0.0011875 --noise 0.3 --freeze-feedback
