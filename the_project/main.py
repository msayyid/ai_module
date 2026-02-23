import numpy as np
import matplotlib.pyplot as plt
import random
def make_env(row, column):
    matrix = np.zeros((row, column))
    return matrix
    # matrix[-1][-1] = 1
    # matrix[3][4] = 1
    # matrix[2][8] = 2
    # plt.imshow(matrix, cmap="binary")
    # plt.show()

def print_environment(environment, agent_pos, number_rep):
    environment[agent_pos] = number_rep["AGENT"]
    print(environment)
    plt.imshow(environment, cmap="binary")
    plt.show()


def set_obstacles(environment, row, column, number_rep):
    # environment is a numpay 2D array, row, column = int
    # create a river in the middle
    for i in range(row):
        environment[row//2][i] = number_rep["RIVER"]

    # set random obstacles
    # print(random_location)
    for i in range(row):
        random_location = random.randint(0, column)

        for j in range(column):
            if i == (row//2):
                continue
            else:
                environment[i][random_location-1] = number_rep["OBSTACLE"]


    return environment

def move(agent_pos, env):
    for y in range(row):
        for x in range(column//2):
            agent_pos = x
    return agent_pos


number_rep = {
    "AGENT": 5,
    "EMPTY": 0,
    "OBSTACLE": 3,
    "RIVER": 4
}
row = 12
column = 12
agent_pos = (0, 0)
the_env = make_env(row, column)
# set_obstacles(the_env, row, column, number_rep)
print_environment(the_env, agent_pos, number_rep)
move(agent_pos, the_env)
print_environment(the_env, move(agent_pos, the_env), number_rep)
print(the_env[row//2][:])