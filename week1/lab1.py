def rps_game(player1, player2):
    if (player1 == "rock" and player2 == "rock") or (player1 == "paper" and player2 == "paper") or (player1 == "scissors" and player2 == "scissors"):
        return "draw"

    elif (player1 == "rock" and player2 == "scissors") or (player1 == "scissors" and player2 == "paper") or (player1 == "paper" and player2 == "rock"):
        return "player 1 wins"
    else: return "player 2 wins"

print("Welcome to the game!")
player2 = ""
player1 = ""
while True:
    
    player1 = input("Player 1: rock/paper/scissors: ").lower()
    player2 = input("Player 2: rock/paper/scissors: ").lower()

    if player1 == "stop" or player2 == "stop":
        print("bye bye")
        break

    print(rps_game(player1, player2))

