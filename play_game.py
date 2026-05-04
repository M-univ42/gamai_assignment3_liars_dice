import game_engine
import bots
import os


num_games = 20


bots_random = [
    bots.random_bot,
    bots.random_bot,
    bots.random_bot,
    bots.random_bot,
]



def log_results(results, filename = "results.txt", folder = "results"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    with open(filepath, "w") as f:
        f.write("game_id,winner_id\n")
        for game_id, winner_id in results:
            f.write(f"{game_id},{winner_id}\n")


if __name__ == "__main__":

    results = []

    # play games
    for i in range(num_games):
        winner_id = game_engine.play_game(bots_random)
        results.append((i, winner_id))
        print("Game ", i ," -- ", " winner :", winner_id)

    # log results
    log_results(results)

