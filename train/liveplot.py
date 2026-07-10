import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        (self.l_train_loss,) = self.ax_loss.plot([], [], label="train")
        (self.l_val_loss,) = self.ax_loss.plot([], [], label="val")
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.grid(True)
        self.ax_loss.legend()

        (self.l_train_acc,) = self.ax_acc.plot([], [], label="train")
        (self.l_val_acc,) = self.ax_acc.plot([], [], label="val")
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.grid(True)
        self.ax_acc.legend()

        self.fig.tight_layout()

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        self.l_train_loss.set_data(self.epochs, self.train_loss)
        self.l_val_loss.set_data(self.epochs, self.val_loss)
        self.l_train_acc.set_data(self.epochs, self.train_acc)
        self.l_val_acc.set_data(self.epochs, self.val_acc)

        for ax in (self.ax_loss, self.ax_acc):
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)