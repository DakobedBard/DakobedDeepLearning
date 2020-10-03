### TMUX 

#### Create a new session
tmux new -s tflow_session

tmux attach -t tflow_session

tmux list-sessions

splits the window into two vertical panes
tmux split-window 

splits the window into two horizontal panes
tmux split-window -h (prefix + ")

selects the next pane in the specified direction
tmux select-pane -[UDLR]

tm

