import tkinter as tk
import tensorflow as tf

class Simulator:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Cellular Automata Simulator")
        self.master.geometry("200x200")

        self.grid = [[0 for _ in range(10)] for _ in range(10)]

        self.canvas = tk.Canvas(self.master, width=200, height=200)
        self.canvas.pack()

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10,10)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def update_grid(self):
        # use the model to update the state of the grid
        updated_grid = self.model.predict(self.grid)
        self.grid = updated_grid

        for row in range(10):
            for col in range(10):
                if self.grid[row][col] == 0:
                    color = "white"
                else:
                    color = "black"
                self.canvas.create_rectangle(row*20, col*20, (row+1)*20, (col+1)*20, fill=color)

    def run(self):
        self.update_grid()
        self.master.after(1000, self.run)

root = tk.Tk()
simulator = Simulator(root)
simulator.run()
root.mainloop()
