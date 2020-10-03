
-L 16006:127.0.0.1:6006
ssh -Y -L 16006:127.0.0.1:6006 darrd@osprey.ocean.washington.edu

start a tmux session
tmux new -s tflow_session

tmux 
tensorboard --logdir=logs 




#### From ipython select the left pane
!tmux select-pane -L  


#### From localhost navigate to tensorboard in browser
http://localhost:16006/  -> tensorboard running here
