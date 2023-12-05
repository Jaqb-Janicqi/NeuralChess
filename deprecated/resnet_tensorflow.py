from keras import Model, Sequential
from keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Flatten, Dense


class ResBlock(Layer):
    def __init__(self, num_features, dtype):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(num_features, 3, padding='same')
        self.conv2 = Conv2D(num_features, 3, padding='same')
        self.bnorm1 = BatchNormalization()
        self.bnorm2 = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x += residual
        x = self.relu(x)
        return x


class ResNet(Model):
    def __init__(self, num_blocks, num_features, num_input_features, policy_size):
        super(ResNet, self).__init__()
        self.start_block = Sequential([
            Conv2D(num_features, 3, padding='same'),
            BatchNormalization(),
            ReLU()
        ])
        self.blocks = [ResBlock(num_features) for _ in range(num_blocks)]
        self.policy = Sequential([
            Conv2D(num_features, 1),
            Conv2D(num_features, 1),
            BatchNormalization(),
            ReLU(),
            Flatten(),
            Dense(policy_size)
        ])
        self.value = Sequential([
            Conv2D(num_features, 1),
            BatchNormalization(),
            ReLU(),
            Flatten(),
            Dense(num_features),
            ReLU(),
            Dense(1, activation='tanh')
        ])


    def call(self, inputs):
        x = self.start_block(inputs)
        for block in self.blocks:
            x = block(x)
        p = self.policy(x)
        v = self.value(x)
        return p, v
