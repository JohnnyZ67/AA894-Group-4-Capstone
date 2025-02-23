from nfl import visuals
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

class Field:

    def __init__(self, width=15, length=8):
        self.width = width
        self.length = length

    def generate_field(self):
        fig, ax = visuals.field(
            yard_numbers=True,
            touchdown_markings=True,
            fifty_yard=False,
            fig_size=(self.width, self.length)
        )
        self.fig = fig
        self.ax = ax

    def add_players(self, play: pd.DataFrame):
        self.ax.scatter(play.x, play.y, c='red', s=15)
        self.ax.set_title(f"Formation: {play.offenseFormation.unique()[0]}")

    def zoomed_formation(self, play: pd.DataFrame, title_suffix: str = ""):
        fig, ax = plt.subplots(1, figsize=(5,5))

        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.axis("off")

        # Field
        field = patches.Rectangle((-25, -25), 50, 50, fc='green')
        ax.add_patch(field)

        center_pos = play[play['position'] == 'C'][['x', 'y']]
        center_x = center_pos["x"].values[0]
        center_y = center_pos["y"].values[0]

        centered_play = play.copy()
        centered_play["x"] = play["x"].subtract(center_x)
        centered_play["y"] = play["y"].subtract(center_y)
        
        ax.scatter(centered_play.x, centered_play.y, c=['red'], s=25)
        ax.plot([0, 0], [-25, 25], color='white', linestyle='-', linewidth=.5)
        ax.set_title(f"Formation: {centered_play.offenseFormation.unique()[0]}{title_suffix}")

        return fig
    
    def flip_formation(self, play: pd.DataFrame):
        tmp = play.copy()
        tmp['x'] = play['x'].multiply(-1)
        tmp['y'] = play['y'].multiply(-1)

        return tmp