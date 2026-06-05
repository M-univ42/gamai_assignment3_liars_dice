import game


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              🎲 LIAR'S DICE Start                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("Play as a human player against AI bots.")
    print("Use the terminal for actions and the live plot for bid analysis.")
    print()


def ask_int(prompt, default, min_value, max_value):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if raw == "":
            return default

        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if min_value <= value <= max_value:
            return value

        print(f"Please enter a number between {min_value} and {max_value}.")


def main():
    print_banner()

    human_player_id = ask_int(
        prompt=f"Choose your player id, 0 to {game.NUM_PLAYERS - 1}",
        default=0,
        min_value=0,
        max_value=game.NUM_PLAYERS - 1,
    )

    input("Press Enter to start the game...")

    try:
        game.play_interactive(human_player_id=human_player_id)
    except KeyboardInterrupt:
        print("\nGame closed. See you next round!")


if __name__ == "__main__":
    main()