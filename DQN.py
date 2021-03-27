import tensorflow as tf

class DQN:
    def __init__(self,model,gamma=0.9,learnging_rate=0.01):
        self.act_dim = model.act_dim
        self.act_seq = model.act_seq
        self.act_model = model.act_model
        self.act_target_model = model.act_target_model
        self.move_model = model.move_model
        self.move_target_model = model.move_target_model
        self.gamma = gamma
        self.lr = learnging_rate
        # --------------------------训练模型--------------------------- # 
        self.act_model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.act_model.loss_func = tf.losses.MeanSquaredError()

        self.move_model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.move_model.loss_func = tf.losses.MeanSquaredError()
        # self.act_model.train_loss = tf.metrics.Mean(name="train_loss")
        # ------------------------------------------------------------ #
        self.act_global_step = 0
        self.move_global_step = 0
        self.update_target_steps = 100  # 每隔200个training steps再把model的参数复制到target_model中

    # train functions for act model
    def act_predict(self, obs):
        """ 使用self.act_model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.act_model.predict(obs)

    def act_train_step(self,action,features,labels):
        """ 训练步骤
        """
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.act_model(features,training=True)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions,indices=enum_action)
            loss = self.act_model.loss_func(labels,pred_action_value)
        gradients = tape.gradient(loss,self.act_model.trainable_variables)
        self.act_model.optimizer.apply_gradients(zip(gradients,self.act_model.trainable_variables))
        # self.act_model.train_loss.update_state(loss)
    def act_train_model(self,action,features,labels,epochs=1):
        """ 训练模型
        """
        for epoch in tf.range(1,epochs+1):
            self.act_train_step(action,features,labels)

    def act_learn(self,obs,actions,reward,next_obs,terminal):
        """ 使用DQN算法更新self.act_model的value网络
        """
        # print('learning')
        # 每隔200个training steps同步一次model和target_model的参数
        if self.act_global_step % self.update_target_steps == 0:
            # print('replace')
            self.act_replace_target()

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.act_target_model.predict(next_obs)
        # print(next_pred_value.shape)
        next_pred_value = next_pred_value.reshape((len(reward), self.act_seq, self.act_dim))
        
        best_v = tf.transpose(tf.reduce_max(next_pred_value,axis=2))
        actions = [[row[i] for row in actions] for i in range(len(actions[0]))]
        for i, acts in enumerate(actions):
            for a in acts:
                a += i * self.act_dim
        for i in range(self.act_seq):
            terminal = tf.cast(terminal,dtype=tf.float32)
            target = reward + self.gamma * (1.0 - terminal) * best_v[i]
            # print('get q')
            # 训练模型
            self.act_train_model(actions[i],obs,target,epochs=1)
        self.act_global_step += 1
        # print('finish')
    def act_replace_target(self):
        '''预测模型权重更新到target模型权重'''
        self.act_target_model.get_layer(name='c1').set_weights(self.act_model.get_layer(name='c1').get_weights())
        self.act_target_model.get_layer(name='c2').set_weights(self.act_model.get_layer(name='c2').get_weights())
        self.act_target_model.get_layer(name='c3').set_weights(self.act_model.get_layer(name='c3').get_weights())
        self.act_target_model.get_layer(name='c4').set_weights(self.act_model.get_layer(name='c4').get_weights())
        self.act_target_model.get_layer(name='b1').set_weights(self.act_model.get_layer(name='b1').get_weights())
        self.act_target_model.get_layer(name='b2').set_weights(self.act_model.get_layer(name='b2').get_weights())
        self.act_target_model.get_layer(name='b3').set_weights(self.act_model.get_layer(name='b3').get_weights())
        self.act_target_model.get_layer(name='p1').set_weights(self.act_model.get_layer(name='p1').get_weights())
        self.act_target_model.get_layer(name='p2').set_weights(self.act_model.get_layer(name='p2').get_weights())
        self.act_target_model.get_layer(name='p3').set_weights(self.act_model.get_layer(name='p3').get_weights())
        self.act_target_model.get_layer(name='f1').set_weights(self.act_model.get_layer(name='f1').get_weights())
        self.act_target_model.get_layer(name='d1').set_weights(self.act_model.get_layer(name='d1').get_weights())
        self.act_target_model.get_layer(name='d2').set_weights(self.act_model.get_layer(name='d2').get_weights())
        self.act_target_model.get_layer(name='d3').set_weights(self.act_model.get_layer(name='d3').get_weights())
        self.act_target_model.get_layer(name='dp1').set_weights(self.act_model.get_layer(name='dp1').get_weights())
        self.act_target_model.get_layer(name='dp2').set_weights(self.act_model.get_layer(name='dp2').get_weights())



    # train functions for move_model

    def move_predict(self, obs):
        """ 使用self.move_model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.move_model.predict(obs)

    def move_train_step(self,action,features,labels):
        """ 训练步骤
        """
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.move_model(features,training=True)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions,indices=enum_action)
            loss = self.move_model.loss_func(labels,pred_action_value)
        gradients = tape.gradient(loss,self.move_model.trainable_variables)
        self.move_model.optimizer.apply_gradients(zip(gradients,self.move_model.trainable_variables))
        # self.move_model.train_loss.update_state(loss)
    def move_train_model(self,action,features,labels,epochs=1):
        """ 训练模型
        """
        for epoch in tf.range(1,epochs+1):
            self.move_train_step(action,features,labels)

    def move_learn(self,obs,action,reward,next_obs,terminal):
        """ 使用DQN算法更新self.move_model的value网络
        """

        # 每隔200个training steps同步一次model和target_model的参数
        if self.move_global_step % self.update_target_steps == 0:
            self.move_replace_target()

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.move_target_model.predict(next_obs)        
        best_v = tf.reduce_max(next_pred_value,axis=1)
        terminal = tf.cast(terminal,dtype=tf.float32)
        target = reward + self.gamma * (1.0 - terminal) * best_v

        self.move_train_model(action[i],obs,target,epochs=1)
        self.move_global_step += 1
        # print('finish')
    def move_replace_target(self):
        '''预测模型权重更新到target模型权重'''
        self.move_target_model.get_layer(name='c1').set_weights(self.move_model.get_layer(name='c1').get_weights())
        self.move_target_model.get_layer(name='c2').set_weights(self.move_model.get_layer(name='c2').get_weights())
        self.move_target_model.get_layer(name='c3').set_weights(self.move_model.get_layer(name='c3').get_weights())
        self.move_target_model.get_layer(name='c4').set_weights(self.move_model.get_layer(name='c4').get_weights())
        self.move_target_model.get_layer(name='b1').set_weights(self.move_model.get_layer(name='b1').get_weights())
        self.move_target_model.get_layer(name='b2').set_weights(self.move_model.get_layer(name='b2').get_weights())
        self.move_target_model.get_layer(name='b3').set_weights(self.move_model.get_layer(name='b3').get_weights())
        self.move_target_model.get_layer(name='p1').set_weights(self.move_model.get_layer(name='p1').get_weights())
        self.move_target_model.get_layer(name='p2').set_weights(self.move_model.get_layer(name='p2').get_weights())
        self.move_target_model.get_layer(name='p3').set_weights(self.move_model.get_layer(name='p3').get_weights())
        self.move_target_model.get_layer(name='f1').set_weights(self.move_model.get_layer(name='f1').get_weights())
        self.move_target_model.get_layer(name='d1').set_weights(self.move_model.get_layer(name='d1').get_weights())
        self.move_target_model.get_layer(name='d2').set_weights(self.move_model.get_layer(name='d2').get_weights())
        self.move_target_model.get_layer(name='d3').set_weights(self.move_model.get_layer(name='d3').get_weights())
        self.move_target_model.get_layer(name='dp1').set_weights(self.move_model.get_layer(name='dp1').get_weights())
        self.move_target_model.get_layer(name='dp2').set_weights(self.move_model.get_layer(name='dp2').get_weights())