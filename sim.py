import numpy as np

class WordleSim:
    def __init__(self, goal):
        # set goal
        self.goal = goal
        # get words
        self.words = []
        with open('words.txt', 'r') as f:
            words = [line.strip() for line in f.readlines()]

    def compare(self, guess):
        # 0 for gray
        # 1 for yellow
        # 2 for green
        if guess == self.goal:
            return [2]*len(guess)
        
        match = [0]*len(guess)
        goal_ = self.goal

        # check for green
        for i in range(len(guess)):
            if guess[i] == goal_[i]:
                match[i] = 2
                goal_ = goal_[:i] + "." + goal_[i+1:]
        
        # check for yellow
        for i in range(len(guess)): 
            if guess[i] in goal_:                   
                match[i] = 1
                j = goal_.index(guess[i])
                goal_ = goal_[:j] + "." + goal_[j+1:]
        
        return match


if __name__ == "__main__":
    sim = WordleSim("foals")
    print(sim.compare("rally"))