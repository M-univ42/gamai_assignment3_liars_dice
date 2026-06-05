import csv
from collections import Counter
import matplotlib.pyplot as plt

filename = "results/results.txt"

winner_counts = Counter()

with open(filename, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        winner_id = int(row["winner_id"])
        winner_counts[winner_id] += 1

# Map player ids to labels
label_map = {
    0: "Statistical",
    1: "Random 1",
    2: "Random 2",
    3: "Random 3",
}

players = [label_map[player_id] for player_id in sorted(winner_counts.keys())]
wins = [winner_counts[player_id] for player_id in sorted(winner_counts.keys())]

plt.figure(figsize=(8, 5))
plt.bar(players, wins)

plt.xlabel("Bot")
plt.ylabel("Number of Wins")
plt.title("Wins per Bot")


plt.tight_layout()
plt.savefig("plots/win_plot.png")