
from Code.Models.





class Network(object):

    def __init__(self, model, checkpoint_path):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.test_losses = []
        self.train_loader = None
        self.test_loader = None

    def load_data(self, x, y):


