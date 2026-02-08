# There are 9 cells arranged in a 3x3 grid. Eight of them are numbered from 1 - 8 and one is
# left blank (in my case, I have numbered it to 0).
# The rule of this game is to swap the blank cell with its adjacent cell and arrange the cells in
# ascending order (with the blank space in the last cell).


from collections import deque

# 1 0 3
# 4 2 5
# 7 8 6

def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0) # find where the 0, blank, is; return the index of the value 0 in the list
    row, col = zero_index // 3, zero_index % 3 #  location of a value in the list; converting index into row and column
    
    # possible moves of the blank

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right

    # dr, dc = delta row, delta column => delta = change
    # change in row, change in column
    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc # changing position
        if 0 <= new_row < 3 and 0 <= new_col < 3: # check if the new position of zero is inside the board
            new_index = new_row * 3 + new_col # we swap index 1 with index 4; in second loop new index is at 4 where the value is 2
            new_state = list(state) # our state is a tuple, we convert it to list to modify, now new_state is state(copy)
            # the next line: SWAP
            # new_state[1], new_state[4] = new_state[4], new_state[1], this is the next line with numbers
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            # before swap: 
            # index:  0 1 2 3 4 5 6 7 8
            # value:  1 0 3 4 2 5 7 8 6

            # after swap:
            # index:  0 1 2 3 4 5 6 7 8
            # value:  1 2 3 4 0 5 7 8 6

            # we convert back to tuple, so it can go to a set visited
            neighbors.append(tuple(new_state))

    return neighbors

# brain of the solution
def solve_puzzle(start, goal):
    # get from start to goal state
    # each thing in the queue is one state we want to explore
    queue = deque([(start, [])]) # queue is a list of future things to explore
    # start = the starting board; [] = empty list (no moves yet)
    # we are at the start board, and we have taken zero steps so far
    # we are storing (start, []) tuple in a list, because deque() expects something iterable to fill itself, tuple is iterable, but we don't want deque to iterate over the tuple's elements
    # so the list is used to make sure the tuple is treated as ONE item, not split apart
    # each item in queue is a pair (current board, path_so_far)

    visited = set([start])
    # visited is a memory of all boards we have already seen
    # visited is a set because it answers the question "have i seen this board before?" in constant time
    # visited ensures we never explore the same board more than once


    while queue: # run until queue is empty
        # popleft() - removes the first item in the queue, returns it; FIFO
        # each item in queue looks like this: (current_board, path_so_far)
        # current = the board we are exploring now; path = how we got there
        current, path = queue.popleft()

        # path = the exact sequence of boards we stepped on to arrive at current (not all boards the algorithm has ever seen)

        if current == goal: # stopping condition
            return path + [current] # path = boards before this one, current = final board
        
        for neighbor in get_neighbors(current): # current board has 3 neighbors, loop runs 3 times
            if neighbor not in visited: # this line is extremely fast because visited is set
                # if i have not seen this board before do the following lines: else continue
                visited.add(neighbor) # expanding the list of seen boards
                queue.append((neighbor, path + [current])) # queue is a list of future things to explore
                # later, explore neighbor, when you do, remember you reached it by going through current, after following path
                # i can reach neighbor by following path + [current]

    return None


start = (1, 0, 3,
         4, 2, 5,
         7, 8, 6)


goal = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

result = solve_puzzle(start, goal)
print(result)

result = solve_puzzle(start, goal)
for state in result:
    print("the current state: ", state)

for state in result:
    print(state[:3])
    print(state[3:6])
    print(state[6:])
    print()
print(result)