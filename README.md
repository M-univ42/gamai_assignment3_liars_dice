<<<<<<< HEAD
# Game AI Assignment 3
## Liar's Dice Agent
=======
# liars_dice
Liar's Dice simulator written in Python

Define your bots under the section ### BOTS ###
Bots are called every turn and are passed the game parameters in a dict with the following structure

     {'players':[
            {'bet':(number_of_die,die_value),'num_die':number_of_player_die},
            {'bet':(number_of_die,die_value),'num_die':number_of_player_die},
            {'bet':None,'num_die':6}
            ],
      'previous_id':previous_player_id,
      'previous_bet':(number_of_die,die_value),
      'current_id':active_player_id,
      'your_id':your_id,
      'your_rolls':[6,5,4,3,2,1],
      'game_over':0
    }


Your bot should return a tuple `('number of dice','value')` to bet, `'bluff'` or `'spot on'`.
Your response is ignored when it is not your turn.
>>>>>>> f1fb4262d5736d61c2a7d691289ba705134046aa
