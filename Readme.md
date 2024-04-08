# RoboHub
RoboHub is a neural network based chess engine designed for over the board play.

# Usage
Communication with the engine is performed in console.

## Interfaces
Limited UCI (Universall Chess Iterface) implementation is available and a custom OTB (Over The Board) interaface is available,
which is much simpler and quicker to use via keyboard.

Selection:
* uci\
    selects uci interface.\
    acknowledged with response 'uciok'
* ocb \
    selects ocb interface.\
    acknowledged with response 'ocbok'

### UCI
* ucinewgame \
    resets the engine search tree

* isready\
    responds 'readyok' when the engine has finished initialization

* position [ fen <fenstring> | startpos ]  moves move1 .... movei\
    allows for passing a position in format of fen string or starting a new game

* debug [ on | off ]\
    toggles debug mode

* go
	* searchmoves move1 .... movei\
		restrict search to specified moves
	* ponder\
		start evaluating positions resulting from the move suggested for the opponent
	* wtime x\
		white has x msec left on the clock
	* btime x\
		black has x msec left on the clock
	* winc x\
		white increment per move in mseconds if x > 0
	* binc x\
		black increment per move in mseconds if x > 0
	* movestogo x\
      		tells the engine how many moves are left until bonus time
	* depth x\
		search x plies only
	* nodes x\
	   search x nodes only
	* movetime x\
		search exactly x mseconds
	* infinite\
		search until the "stop" command
    
* stop\
	stop calculation

* ponderhit\
	the user has played the expected move. The engine will continue calculation in normal mode.

* quit\
	quit the program

### OTB
* new \
    resets the engine search tree and creates a default board

* reset\
    resets the engine search tree

* isready\
    responds 'readyok' when the engine has finished initialization

* position [fen <fenstring> | startpos ]  moves move1 .... movei\
    allows for passing a position in format of fen string or starting a new game

* go
	* movetime \
		search exactly x seconds.\
        If movetime is not given, the engine will adjust search time accordingly to total time it has left.

* setoption name [value x]\
	used to change the internal parameters.
    * autopush\
        automatically push the best move after finishing search.
    * autoponder\
        automatically enter ponder search after pushing the best move.
        autopush has to be enabled.

* push move\
    push a move on the board.\
    e.g. push e2e4

## Dependencies
[Microsoft DirectML](https://python-chess.readthedocs.io/en/latest/)\
[PyTorch](https://python-chess.readthedocs.io/en/latest/)\
[NumPy](https://python-chess.readthedocs.io/en/latest/)\
[python-chess](https://python-chess.readthedocs.io/en/latest/)\
[PyYAML](https://pypi.org/project/PyYAML/)
