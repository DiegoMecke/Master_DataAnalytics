#Import packages
import tensorflow as tf

#Define train class
class MyTrainModel:

    def __init__(self, model, batch_size=8, lr=0.001, loss=tf.keras.losses.MeanSquaredError, opt=tf.keras.optimizers.Adam,\
                clip_value=1.0):

        self.model      = model
        self.loss       = loss()
        self.optimizer  = opt(learning_rate=lr, clipvalue=clip_value)
        self.batch_size = batch_size

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.test_loss  = tf.keras.metrics.Mean(name='test_loss')


    @tf.function
    def train_step(self, x , y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        self.train_loss.update_state(loss)
        return loss

    @tf.function
    def test_step(self, x , y):
        predictions = self.model(x)
        loss = self.loss(y, predictions)
        self.test_loss.update_state(loss)
        return loss

    def train(self):
        loss = []
        for bX, bY in self.train_ds:
            loss.append(self.train_step(bX, bY))
        return loss

    def test(self):
        loss = []
        for bX, bY in self.test_ds:
            loss.append(self.test_step(bX, bY))  
        return loss 

    def run(self, train_gen, test_gen, epochs, verbose=2):
        history = []

        for i in range(epochs):
            
            self.train_ds = tf.data.Dataset.from_tensor_slices(next(train_gen)).batch(self.batch_size) #Update
            self.test_ds  = tf.data.Dataset.from_tensor_slices(next(test_gen)).batch(self.batch_size)

            train_loss = self.train()
            test_loss  = self.test()

            history.append([train_loss,test_loss])

            if verbose > 0 and (i==0 or (i+1)%10==0):
                print(f"epoch: {i+1}, TRAIN LOSS: {self.train_loss.result()}, TEST LOSS: {self.test_loss.result()}")

                self.train_loss.reset_states()
                self.test_loss.reset_states()

        return history