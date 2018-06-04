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
import numpy as np
import os
import pdb
from queue import Queue
from threading import Thread
#from multiprocessing import Process as Thread

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

    def __init__(self, Q_experience, *args):
        
        super(Trainer, self).__init__(*args)
        
        self.agent = TrainerAgent(self._game, **self._kwargs_agent)
        
        self.Q_experience = Q_experience

    def run(self, coord, num_episodes='default', num_timesteps='default'):
        pass

    def train(self, coord, num_episodes='default', num_timesteps='default'):
        
        global_step = tf.train.get_or_create_global_step()  

        for epi in range(num_episodes):
            episode = self.Q_experience.get()

            self.agent.train(episode)
            
            self.log.info("global_step: {0}/{1}".format(global_step.numpy(), epi))
            global_step.assign_add(1)
            
        coord.request_stop()

    def eval(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass


class Runner(AbstractRunner):

    def __init__(self, Q_experience, trainer_agent, *args, presenter=False):
        
        super(Runner, self).__init__(*args)
        
        self.agent = RunnerAgent(trainer_agent,
                                 self._game,
                                 **self._kwargs_agent,
                                 presenter=presenter)
        
        self.Q_experience = Q_experience

    def run(self, coord=None, num_episodes='default', num_timesteps='default'):
        if coord:
            while not coord.should_stop():
                episode = self.agent.run(num_timesteps)
                self.Q_experience.put(episode)
        else:
            for _ in range(num_episodes):
                self.agent.run(num_timesteps)
        
        self.agent.env.env.close()


    def train(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass

    def eval(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass


class Manager(AbstractManager):
    def __init__(self, *args):
        
        super(Manager, self).__init__(*args)
        
        # TODO: queue size has to be adjusted according to fill and remove rates
        self.Q_experiences = Queue(maxsize=100) 
        
        if self._mode == 'train':
            self.trainer = Trainer(self.Q_experiences, 
                                   self._algo,
                                   self._game,
                                   self._kwargs_agent)
            
            self.num_runners = 1
            
            self.runners = []
            
            for _ in range(self.num_runners):
                self.runners.append(Runner(self.Q_experiences, 
                                           self.trainer.agent,
                                           self._algo,
                                           self._game,
                                           self._kwargs_agent))
                
            self.presenter = Runner(self.Q_experiences, 
                                    self.trainer.agent,
                                    self._algo,
                                    self._game,
                                    self._kwargs_agent,
                                    presenter=True)
                
        else:
            self.presenter = Runner(self.Q_experiences, 
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
            
            presenter_thread = self.presenter.thread(coord, 'run', num_episodes, num_timesteps)
            
            coord.register_thread(trainer_thread)
            
            for runner_thread in runner_threads:
                coord.register_thread(runner_thread)
            
            coord.register_thread(presenter_thread)
            
            # Start threads
            trainer_thread.start()
            
            for runner_thread in runner_threads:
                runner_thread.start()
            
            presenter_thread.start()
            
            self.log.info('All threads started')
            
            # Join threads
            coord.join(stop_grace_period_secs=Config.TF_THREAD_COORDINATOR_GRACE_TIME_SEC)
            
            coord.wait_for_stop(timeout=10)
            
            self.log.info('All threads finished')
            
        elif mode == 'eval':
            
            self.log.info('Evaluation started')
            
            self.presenter.eval(num_episodes=num_episodes,
                                num_timesteps=num_timesteps)
            
            self.log.info('Evaluation finished.')
            
        elif mode == 'run':
            
            self.log.info('Run started')
            
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
        
        self._input_shape = self._state_shape
        
        self.share_weights = share_weights
        
        self._presenter = presenter
        
        self.log = Logger.get_active_logger()
        
        if self._frame_based_states:
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
        
        if self._presenter:
            
            # Tensorboard 
            self._tb_summary = tf.contrib.summary
            
            self._tb_path = os.path.join(Config.TENSORBOARD_PATH, self._game, RL_ALGORITHM) 
            
            self._tb_record_intervall = Config.TENSORBOARD_RECORD_INTERVALL
            
            self._tb_writer = tf.contrib.summary.create_file_writer(self._tb_path)
            
            self._tb_writer.set_as_default()
            
            # Checkpoints
            path_to_this_file = os.path.abspath(os.path.dirname(__file__))
            
            self._checkpoint_dir = os.path.join(path_to_this_file, 
                                                checkpoint_dir, 
                                                self._game,
                                                RL_ALGORITHM)
            
            self._checkpoint = tfe.Checkpoint(policy=Policy(self._base_net_policy,
                                                            self._action_dim))
            
        self._agent_initialized = False

    def init_agent(self):
        
        state = tf.random_normal(self._input_shape)
        
        state = state[np.newaxis, :]
        
        if self._actionspace_is_discrete:
            
            action_probs = self._policy(state)
            
            action = np.argmax(action_probs)
        
        else:
            mu, sigma = self._policy(state)
        
            self.action_dist = self.ActionDist(tf.squeeze(mu), tf.squeeze(sigma))
        
        value = self._valuef(state)

    def run(self, num_timesteps):
        
        if self._agent_initialized is not True:
            self._agent_initialized = True
            
            self.init_agent()
        
        self.log.info("Enter Agent '{}' run".format(self.name))
        
        state = self.env.reset()
        
        state = self.to_tf_tensor(state[np.newaxis, :])
        
        episode = {'actions': [], 'action_probs': [], 'values': [], 'states': [],
                   'rewards':[], 'target_values': []}
        
        with tfe.GradientTape() as g:
            g.watch([self._policy.variables, self._valuef.variables])
            
            for step in range(num_timesteps):
                if self._presenter:
                    self.env.render()
                
                if self._actionspace_is_discrete:
                    action_probs = self._policy(state)
                    
                    action = np.argmax(action_probs)
                    
                    state_, reward, done, __ = self.env.step(action)
                    
                    episode['actions'].append(action)
                
                else:
                    mu, sigma = self._policy(state)
                    
                    self.action_dist = self.ActionDist(tf.squeeze(mu), tf.squeeze(sigma))
                    
                    action = self.action_dist.sample(1)
                    
                    action_probs = self.action_dist.prob(action)
                    
                    state_, reward, done, __ = self.env.step(action)
                    
                    episode['actions'].append(action.numpy())
                
                episode['action_probs'].append(action_probs.numpy())
                
                value = self._valuef(state)
                
                if self._tensorboard and self._presenter:
                    
                    with tf.contrib.summary.record_summaries_every_n_global_steps(self._tb_record_intervall):
                        self._tb_summary.scalar('value', value)
                    
                    if done or step == num_timesteps-1:
                        self._tb_summary.flush()
                        
                if self._presenter and (done or step == num_timesteps-1):
                    
                    self._checkpoint.save(self._checkpoint_dir)
                    
                episode['values'].append(value.numpy())
                
                episode['states'].append(state.numpy())
                
                episode['rewards'].append(reward)
                
                state = self.to_tf_tensor(state_[np.newaxis, :])
                
                if done:
                    break

        episode_ = {key: np.array(val) for key, val in episode.items()}
        
        episode_['target_values'] = self.discounted_target_values(episode['rewards'],
                                                                     episode['values'][-1])
        # Losses
        # TODO losses and gradient undefined (inf, nan) due bad nets 
#        self._loss_valuef  =   tf.reduce_sum(tf.square(episode_['target_values'] - episode_['values']))
#        
#        self._loss_entropy = - tf.reduce_sum(episode_['action_probs'] * tf.log(episode_['action_probs']))
#        
#        self._loss_policy  = - tf.reduce_sum(tf.log(episode_['action_probs']))
#        
#        self._loss_total =  0.5 * self._loss_valuef # self._loss_policy - self._loss_entropy * 0.01 
#        
#        if self._agent_initialized:
#            grad_policy, grad_valuef = g.gradient(self._loss_total, [self._policy.variables, self._valuef.variables])
#            grad_valuef = g.gradient(self._loss_total, self._valuef.variables)
#        pdb.set_trace()
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#        optimizer.apply_gradients(zip(grad_policy, self._policy.variables))
#        optimizer.apply_gradients(zip(grad_valuef, self._valuef.variables))
#        return {'grad_valuef': grad_valuef, 'grad_policy': grad_policy}
        
        self.env.env.reset()
        return [episode_['actions'], episode_['states'], episode_['target_values']]

    def set_env_info(self, env):
        
        self._actionspace_is_discrete = env.action_space.__class__ == gym.spaces.discrete.Discrete
        
        self._frame_based_states   = len(env.observation_space.shape) == 3
        
        if env.action_space.shape:
            self._action_dim = env.action_space.shape[0]
            
        else:
            self._action_dim = env.action_space.n
            
        self._state_shape = list(env.observation_space.shape)

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


class TrainerAgent(BaseAgent):
    
    def __init__(self, *args, **kwargs):
        
        super(TrainerAgent, self).__init__(*args, **kwargs)
        
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    def train(self, episode):
        
        self.log.info('Training ongoing ...')
#        TODO requires well behaved gradients - bad nets
#        self.optimizer.apply_gradients(zip(episode['grad_policy'], self._policy.variables))
#        self.optimizer.apply_gradients(zip(episode['grad_valuef'], self._valuef.variables))


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
