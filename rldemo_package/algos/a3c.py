#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    MIT License
#    
#    Copyright (c) 2018 Timo Wildhage
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

# Build-in modules
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import os
import pdb
from queue import Queue
import sys
from threading import Thread

# Third party modules 
import gym
from tensorflow import keras
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Custom modules
from base_nets import DenseBaseNet, ConvBaseNet
from config import Config
from manager import AbstractManager
from runner import AbstractRunner
from trainer import AbstractTrainer
from utils import map_game_to_net, Logger


RL_ALGORITHM = __file__.split('/')[-1].split('.')[0]


class Trainer(AbstractTrainer):

    def __init__(self, Q_episodes, *args):
        
        super(Trainer, self).__init__(*args)
        
        self.agent = TrainerAgent(self._game, **self._kwargs_agent)
        
        self.Q_episodes = Q_episodes

    def run(self, coord, num_episodes='default', num_timesteps='default'):
        pass

    def train(self, coord, num_episodes='default', num_timesteps='default'):
        
        global_step = tf.train.get_or_create_global_step()  
        self.replay_buffer = []
        
        for epi in range(num_episodes):
            
            episode = self.Q_episodes.get()
            
            self.replay_buffer.append(episode)
            
            for _epi in self.replay_buffer:
                self.agent.train(_epi)
                
                self.agent.log_train(num_episodes, current_episode=epi)
            
            global_step.assign_add(1)
            
        coord.request_stop()

    def eval(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass


class Runner(AbstractRunner):

    def __init__(self, Q_episodes, trainer_agent, *args, presenter=False):
        
        super(Runner, self).__init__(*args)
        
        self.agent = RunnerAgent(trainer_agent,
                                 self._game,
                                 **self._kwargs_agent,
                                 presenter=presenter)
        
        self.Q_episodes = Q_episodes

    def run(self, coord=None, num_episodes='default', num_timesteps='default'):
        
        for epi in range(num_episodes):
            
            episode = self.agent.run(num_timesteps)
            
            self.Q_episodes.put(episode)
            
            if coord:
                if coord.should_stop():
                    break
        
        self.agent.env.env.close()


    def train(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass

    def eval(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass


class Manager(AbstractManager):
    def __init__(self, *args):
        
        super(Manager, self).__init__(*args)
        
        self.Q_episodess = Queue(maxsize=100) 
        
        self.num_runners = self.num_threads
        
        if self._mode == 'train':
            self.trainer = Trainer(self.Q_episodess, 
                                   self._algo,
                                   self._game,
                                   self._kwargs_agent)
            
            self.runners = []
            
            for _ in range(self.num_runners):
                self.runners.append(Runner(self.Q_episodess, 
                                           self.trainer.agent,
                                           self._algo,
                                           self._game,
                                           self._kwargs_agent))
            
            self.presenter = Runner(self.Q_episodess, 
                                    self.trainer.agent,
                                    self._algo,
                                    self._game,
                                    self._kwargs_agent,
                                    presenter=True)
                
        else:
            self.presenter = Runner(self.Q_episodess, 
                                    [],
                                    self._algo,
                                    self._game,
                                    self._kwargs_agent,
                                    presenter=True)


    def run(self, num_episodes, num_timesteps):
        
        coord = tf.train.Coordinator()
        
        mode = self._mode
        
        if mode == 'train':
            
            self.trainer.agent.init_agent()
            
            # Get threads - the trainer stops all other threads after 'num_epiodes'
            trainer_thread = self.trainer.thread(coord, 'train', num_episodes)
            
            runner_threads = [runner.thread(coord, 'run', num_episodes, num_timesteps) for runner in self.runners]
            
            coord.register_thread(trainer_thread)
            
            for runner_thread in runner_threads:
                coord.register_thread(runner_thread)
            
            # Start threads
            trainer_thread.start()
            
            for runner_thread in runner_threads:
                runner_thread.start()
            
            # Add, register and start presenter thread if render flag set
            if self._render:
                presenter_thread = self.presenter.thread(coord, 'run', num_episodes, num_timesteps)
                
                coord.register_thread(presenter_thread)
                
                presenter_thread.start()
            
            self.log.info('All threads started')
            
            # Join threads
            coord.join(stop_grace_period_secs=Config.TF_THREAD_COORDINATOR_GRACE_TIME_SEC)
            
            coord.wait_for_stop(timeout=10)
            
            self.log.info('All threads finished')
            
            self.log.info('Save checkpoint...')
            
            tfe.Saver(self.trainer.agent.variables).save(self.trainer.agent._checkpoint_dir,
                                         global_step=num_episodes)
            
            # Plot losses
            plt.plot(self.trainer.agent.history['loss'])
            
            plt.show()
            
        elif mode == 'eval':
            
            self.log.info('Evaluation started')
            
            self.presenter.eval(num_episodes=num_episodes,
                                num_timesteps=num_timesteps)
            
            self.log.info('Evaluation finished.')
            
        elif mode == 'run':
            
            self.log.info('Run started')
            
            if self._render == False:
                self.presenter.agent._presenter=False
            
            self.presenter.run(num_episodes=num_episodes,
                               num_timesteps=num_timesteps)
            
            self.log.info('Run finished.')


    def summary(self):
        pass


class Policy(keras.Model):

    def __init__(self, base_net, action_dim, discrete_actions=True):
        
        super(Policy, self).__init__()
        
        self._base_net = base_net
        
        self._discrete_actions = discrete_actions
        
        if discrete_actions:
            self._layer_out = keras.layers.Dense(action_dim, 
                                                 activation=tf.keras.activations.softmax)
            
        else:
            self._layer_mu_out = tf.layers.Dense(action_dim)
            
            self._layer_sigma_out = tf.layers.Dense(action_dim)
            

    def call(self, input):
        
        out = self._base_net(input)
        
        if self._discrete_actions:
            probs = self._layer_out(out)
            
            return probs
        
        else:
            mu = self._layer_mu_out(out)
            
            sigma = self._layer_sigma_out(out)

            return mu, sigma


class ValueFunction(keras.Model):

    def __init__(self, base_net):
        
        super(ValueFunction, self).__init__()
        
        self._base_net = base_net
        
        self._layer_out = keras.layers.Dense(1)
        
        
    def call(self, input):
        
        out = self._base_net(input)
        
        out = self._layer_out(out)
        
        return out


class BaseAgent(keras.Model):

    def __init__(self, 
                 game,
                 tensorboard,
                 checkpoint_dir,
                 share_weights,
                 presenter=False,
                 name='A3CAgent'):
        
        super(BaseAgent, self).__init__(name=name)
        
        self._game = game
        
        self.env = gym.make(game)
        
        self.set_env_info(self.env)
        
        self._tensorboard = tensorboard
        
        self._input_shape = self._observation_shape
        
        self.share_weights = share_weights
        
        self._presenter = presenter
        
        self.log = Logger.get_active_logger()
        
        if self._frame_based_observations:
            self._BaseNet = ConvBaseNet
        
        else:
            self._BaseNet = DenseBaseNet
        
        if share_weights:
            self._base_net = self._BaseNet(self._input_shape)
            
            self._policy = Policy(self._base_net, self._action_dim,
                                  self._actionspace_is_discrete) 
            
            self._valuef = ValueFunction(self._base_net) 
        
        else:
            self._base_net_policy = self._BaseNet(self._input_shape)
            
            self._policy = Policy(self._base_net_policy, self._action_dim,
                                  self._actionspace_is_discrete)
            
            self._base_net_valuef = self._BaseNet(self._input_shape)
            
            self._valuef = ValueFunction(self._base_net_valuef)
        
        self.discount_factor = 0.95
        
        self.discount_factors = []
        
        self.ActionDist = tf.contrib.distributions.Normal
        
        
        self.history = {'loss': []}
        
        if self._presenter:
            
            # Tensorboard 
            self._tb_summary = tf.contrib.summary
            
            self._tb_path = os.path.join(Config.TENSORBOARD_PATH, self._game, RL_ALGORITHM) 
            
            self._tb_record_intervall = Config.TENSORBOARD_RECORD_INTERVALL
            
            self._tb_writer = tf.contrib.summary.create_file_writer(self._tb_path)
            
            self._tb_writer.set_as_default()
            
        # Checkpoints
        # TODO fix conditional checkpoint for shared net
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        
        self._checkpoint_dir = os.path.join(path_to_this_file, 
                                            checkpoint_dir, 
                                            self._game,
                                            RL_ALGORITHM)
        
        self._checkpoint = []
            
        self._agent_initialized = False

    def _normalize_observation(self, observation):
        
        if self._frame_based_observations:
            s = observation/255 - 0.5
            
        else:
            # TODO add reasonable values for continuous observation normalization
            s = observation/10 - 5
            
        return s

    def init_agent(self):
        
        with tfe.restore_variables_on_create(
            tf.train.latest_checkpoint(self._checkpoint_dir)):
        
        
            observation = tf.random_normal(self._input_shape)
            
            observation = observation[np.newaxis, :]
            
            if self._actionspace_is_discrete:
                
                action_probs = self._policy(observation)
                
                action = np.argmax(action_probs)
            
            else:
                mu, sigma = self._policy(observation)
            
                self.action_dist = self.ActionDist(tf.squeeze(mu), tf.squeeze(sigma))
            
            value = self._valuef(observation)

    def run(self, num_timesteps):
        
        if self._agent_initialized is not True:
            self._agent_initialized = True
            
            self.init_agent()
        
        observation = self.env.reset()
        
        observation = self.to_tf_tensor(observation[np.newaxis, :])
        
        episode = {'actions': [], 'action_probs': [], 'values': [], 'observations': [],
                   'rewards':[], 'target_values': []}
        
        for step in range(num_timesteps):
            
            observation = self._normalize_observation(observation)
            
            if self._presenter:
                self.env.render()
            
            if self._actionspace_is_discrete:
                if np.random.rand() > 0.9:
                    action_probs = self._policy(observation)
                else:
                    action_probs = tf.convert_to_tensor(keras.utils.to_categorical(tf.random_uniform([self._action_dim]), self._action_dim))
                action = np.argmax(action_probs)
                
                observation_, reward, done, __ = self.env.step(action)
                
                episode['actions'].append(action)
            
            else:
                mu, sigma = self._policy(observation)
                
                self.action_dist = self.ActionDist(tf.squeeze(mu), tf.squeeze(sigma))
                
                action = self.action_dist.sample(1)
                
                action_probs = self.action_dist.prob(action)
                
                observation_, reward, done, __ = self.env.step(action)
                
                episode['actions'].append(action.numpy())
            
            episode['action_probs'].append(action_probs.numpy())
            
            value = self._valuef(observation)
            
            if self._tensorboard and self._presenter:
                
                with tf.contrib.summary.record_summaries_every_n_global_steps(self._tb_record_intervall):
                    self._tb_summary.scalar('value', value)
                
                if done or step == num_timesteps-1:
                    self._tb_summary.flush()
                    
            if self._presenter and (done or step == num_timesteps-1):
                pass
#            TODO fix checkpoint conditioned of shared net
#                self._checkpoint.save(self._checkpoint_dir)
                
            episode['values'].append(value.numpy())
            
            episode['observations'].append(observation.numpy())
            
            episode['rewards'].append(reward)
            
            observation = self.to_tf_tensor(observation_[np.newaxis, :])
            
            if done:
                break

        episode_ = {key: np.squeeze(np.array(val)) for key, val in episode.items()}
        
        episode_['target_values'] = self.discounted_target_values(episode['rewards'],
                                                                     episode['values'][-1])
        
        self.env.env.reset()
        
        return episode_


    def compute_gradients(self, episode):
        
        with tf.GradientTape() as tape:
            self._loss_valuef  =  tf.reduce_mean(tf.square(episode['target_values'] - self._valuef(episode['observations'])))
            
#            self._loss_entropy = - tf.reduce_mean(self._policy(episode['observations']) * tf.log(self._policy(episode['observations'])))
            
            self._loss_policy  = - tf.reduce_mean(tf.log(self._policy(episode['observations'])))
            
            self._loss_reward  = - 10*tf.reduce_mean(self._valuef(episode['observations'])) #tf.constant(0.)
            
            self._loss_total   =  (self._loss_valuef * 0.5 + self._loss_policy + self._loss_reward )#+ self._loss_entropy)
            
        return tape.gradient(self._loss_total, [self._valuef.variables, self._policy.variables])


    def apply_gradients(self, model, optimizer, gradients):
        optimizer.apply_gradients(zip(gradients, model.variables))


    def set_env_info(self, env):
        
        self._actionspace_is_discrete = env.action_space.__class__ == gym.spaces.discrete.Discrete
        
        self._frame_based_observations   = len(env.observation_space.shape) == 3
        
        if env.action_space.shape:
            self._action_dim = env.action_space.shape[0]
            
        else:
            self._action_dim = env.action_space.n
            
        self._observation_shape = list(env.observation_space.shape)


    def to_tf_tensor(self, np_array):
        
        return tf.convert_to_tensor(np_array, np.float32)


    def discounted_target_values(self, rewards, final_value):
        
        n = len(rewards)
        
        if len(self.discount_factors) is not n:
            self.discount_factors = (np.ones((n, )) * self.discount_factor)**(np.arange(n)[::-1])
        
        rewards = np.array(rewards)
        
        dfs = self.discount_factors
        
        target_values = np.ones((n, ))
        
        for i in range(0, n):
            target_values[i] = final_value * dfs[i] +  np.sum(rewards[i::]*dfs[i::])
        
        return target_values


    def log_train(self, num_episodes, current_episode):
        
        episode_info = "episode: {0}/{1}".format(num_episodes, current_episode)
        
        self.log.info("{0}, losses: lv: {1:4.4f},\t lp: {2:4.4f},\t lr: {3:4.4f},\t lt: {4:4.4f}".format(episode_info,
                                                                  self._loss_valuef.numpy(),
                                                                  self._loss_policy.numpy(),
                                                                  self._loss_reward.numpy(),
                                                                  self._loss_total.numpy()))


class TrainerAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        
        super(TrainerAgent, self).__init__(*args, **kwargs)
        
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


    def train(self, episode):
        
        grad_valuef, grad_policy = self.compute_gradients(episode)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
        optimizer.apply_gradients(zip(grad_policy, self._policy.variables))
        optimizer.apply_gradients(zip(grad_valuef, self._valuef.variables))
        
        if self._loss_total > 1e7 or self._loss_total is np.nan:
            self.log.warning('loss exploded, abort training... ')

            sys.exit()
        
        self.history['loss'].append(self._loss_total)


class RunnerAgent(BaseAgent):

   
    def __init__(self, trainer_agent, *args, **kwargs):
        
        super(RunnerAgent, self).__init__(*args, **kwargs)
        
        self.trainer_agent = trainer_agent

    def eval(self, num_timesteps):
        self.log.info('Enter Runner eval, do nothing yet')


    def policy_op(self, state):
        
        return self._policy(state)

    def valuef_op(self, state):
        
        return self._valuef(state)
    
    def eval(self, num_timesteps):
        pass
