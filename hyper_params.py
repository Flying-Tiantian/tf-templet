class hyparams:
    def __init__(self):
        self.moving_average_decay = 0.9999  # The decay to use for the moving average.
        self.init_lr = 0.001  # Initial learning rate.
        self.lr_decay = 0.5  # Learning rate decay factor.
        self.epoch_num_per_decay = 20.0  # Epochs after which learning rate decays.
        self.max_epochs = 500
        self.batch_size = 8
        self.test_batch_size = 1
        self.save_checkpoint_interval = 5
        self.save_summary_interval = 5
        self.test_interval = 1
        self.max_to_keep = 10
        self.weight_decay = None

    def data_info(self, reader):
        self.num_classes = reader.class_num()
        self.train_example_num = reader.example_num_for_train()
        self.test_example_num = reader.example_num_for_eval()
        self.decay_steps = int(self.train_example_num / self.batch_size * self.epoch_num_per_decay)
        self.max_steps = int(self.train_example_num / self.batch_size * self.max_epochs)
        self.epoch_steps = int(self.train_example_num / self.batch_size)
        self.test_steps = int(self.test_example_num / self.test_batch_size)
        self.test_interval *= self.epoch_steps
        self.save_checkpoint_interval *= self.epoch_steps
        self.save_summary_interval = int(self.epoch_steps / self.save_summary_interval)
