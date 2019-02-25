import pygame
import random
import time
import threading
from tetris_env import Env
from PIL import Image
from PIL import ImageOps

from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
import tensorflow as tf

global episode
episode = 0
EPISODES = 800000

def pre_processing(observe):
    processed_observe = np.maximum(observe,observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (40, 40), mode='constant') * 255)
    return processed_observe

class A3CAgent:
    def __init__(self):
        self.threads = 8 # 액터러너의 갯수
        self.state_size = (40, 40, 4)  # (rows, cols, num)

        # [1: Left, 2: Right, 3:Rotate, 5:stop]
        self.action_space = [1, 2, 3, 5]
        self.action_size = 4

        # 글로벌 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()

        ############### 아직 체크 못한것 ##############
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4

        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)
        #########################################

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def train(self):
        agents = []
        agents.append(deepAgent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer],True))
        agents.append(deepAgent(self.action_size, self.state_size,
                                [self.actor, self.critic], self.sess,
                                self.optimizer, self.discount_factor,
                                [self.summary_op, self.summary_placeholders,
                                 self.update_ops, self.summary_writer], False))
        agents.append(deepAgent(self.action_size, self.state_size,
                                [self.actor, self.critic], self.sess,
                                self.optimizer, self.discount_factor,
                                [self.summary_op, self.summary_placeholders,
                                 self.update_ops, self.summary_writer], False))
        agents.append(deepAgent(self.action_size, self.state_size,
                                [self.actor, self.critic], self.sess,
                                self.optimizer, self.discount_factor,
                                [self.summary_op, self.summary_placeholders,
                                 self.update_ops, self.summary_writer], False))

        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(60*10)
            self.save_model("./save_model/breakout_a3c")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def build_model(self):

        input = Input(self.state_size)
        #############  CNN  #################
        conv = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input)  # output = 7x14x16
        conv = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(conv)  # output = 5x10x32
        conv = Flatten()(conv)  # output = 1600
        fc = Dense(256, activation='relu')(conv)
        #####################################

        policy = Dense(self.action_size,activation='softmax')(fc)
        value = Dense(1,activation='linear')(fc)

        # actor: 상태를 받아 각 행동의 확률을 계산
        actor = Model(inputs=input,outputs=policy)

        # critic: 상태를 받아서 상태의 가치를 계산
        critic = Model(inputs=input,outputs=value)

        # 멀티스레딩을 케라스에서 이용할 때 발셍하는 에러 제거
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor,critic

class deepAgent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops,render):
        threading.Thread.__init__(self)

        self.t_max = 20
        self.t = 0

        # rendering
        self.render = render
        self.playMode = False

        # 글로벌 신경망 actor, critic (상속)
        self.actor, self.critic = model

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()
        self.avg_p_max = 0
        self.avg_loss = 0

    def run(self):
        global episode
        env = Env(self.render,self.playMode)
        step = 0

        while episode < EPISODES:
            done = False
            score = 0

            observe = env.reset()
            next_observe = observe

            # 0 ~ 5 frame동안 아무것도 하지 않음.
            for _ in range(random.randint(1,5)):
                observe = next_observe
                next_observe, _, _ = env.step(5)

            # image를 pre_processing
            state = pre_processing(next_observe)
            # 처음엔 4개 동일한 이미지를 history로 작성
            history = np.stack((state,state,state,state),axis=2)
            history = np.reshape([history],(1,self.state_size[0],self.state_size[1],4))

            # One Episode
            while not done:
                step += 1
                self.t += 1
                action, policy = self.get_action(history)

                # policy에 따라 action 선택
                # 1: Left, 2: Right, 3:Rotate, 5:stop
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                elif action == 2:
                    real_action = 3
                else:
                    real_action = 5

                # 전처리
                next_observe,reward,done = env.step(real_action)
                next_state = pre_processing(next_observe)
                next_state = np.reshape([next_state],(1,self.state_size[0],self.state_size[1],1))
                next_history = np.append(next_state,history[:,:,:,:3],axis=3)

                # 정책의 최댓값
                self.avg_p_max += np.amax(self.actor.predict(np.float32(history/255.)))

                score += reward
                reward = np.clip(reward,-1.,1.)

                self.append_sample(history,action,reward)

                # next_history로 다음 action 결정
                history = next_history

                # 최대 타임 스텝수 도달 or 에피소드 종료시 학습
                if self.t >= self.t_max or done:

                    # append_sample 해둔 것을 토대로 글로벌 신경망 업데이트
                    self.train_model(done)

                    # 글로벌 신경망으로 액터러너 업데이트
                    self.update_local_model()
                    self.t = 0

                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:", step)

                    stats = [score, self.avg_p_max / float(step), step]

                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0


    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.local_critic.predict(np.float32(
                self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 글로벌신경망 업데이트
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), self.state_size[0], self.state_size[1], 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.local_critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택 (0,1,2,3)
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        input = Input(shape=self.state_size)
        #############  CNN  #################
        conv = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input)  # output = 7x14x16
        conv = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(conv)  # output = 5x10x32
        conv = Flatten()(conv)  # output = 1600
        fc = Dense(256, activation='relu')(conv)
        #####################################

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        # actor: 상태를 받아 각 행동의 확률을 계산
        # critic: 상태를 받아서 상태의 가치를 계산
        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()