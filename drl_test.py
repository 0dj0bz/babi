import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


class ProbabilityDistribution(tf.keras.Model):
	def call(self, logits, **kwargs):
		return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
	def __init__(self, num_actions):
		super().__init__('mlp_policy')
		self.hidden1 = kl.Dense(128, activation='relu')
		self.hidden2 = kl.Dense(128, activation='relu')
		self.value = kl.Dense(1, name='value')
		self.logits = kl.Dense(num_actions, name='policy_logits')
		self.dist = ProbabilityDistribution()


	def call(self, inputs, **kwargs):
		x = tf.convert_to_tensor(inputs)

		hidden_logs = self.hidden1(x)
		hidden_vals = self.hidden2(x)
		return self.logits(hidden_logs), self.value(hidden_vals)

	def action_value(self, obs):
		logits, value = self.predict_on_batch(obs)
		action = self.dist.predict_on_batch(logits)
		return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class A2CAgent:
	def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
		self.gamma = gamma
		self.value_c = value_c
		self.entropy_c = entropy_c

		self.model = model

		self.model.compile(
			optimizer=ko.RMSprop(lr=lr),
			loss=[self._logits_loss, self._value_loss])

	def train(self, env, batch_sz=64, updates=250):
		actions = np.empty((batch_sz,), dtype=np.int32)
		rewards, dones, values = np.empty((3, batch_sz))
		observations = np.empty((batch_sz,) + env.observation_space.shape)

		ep_rewards = [0,0]
		next_obs = env.reset()

		for update in range(updates):
			for step in range(batch_sz):
				observations[step] = next_obs.copy()
				actions[step], values[step] = self.model.action_value(next_obs[None, :])
				next_obs, rewards[step], dones[step], _ = env.step(actions[step])

				ep_rewards[-1]  += rewards[step]

				if dones[step]:
					ep_rewards.append(0.0)
					next_obs = env.reset()
					logging.info("Episode: %03d, Reward:  %03d" % (len(ep_rewards)-1, ep_rewards[-2]))

			_, next_value = self.model.action_value(next_obs[None, :])

			returns, advs = self._returns_advantages(rewards, dones, values, next_value)

			acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

			losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

			logging.debug("[%d/%d] Losses: %s" % (update+1, updates, losses))

		return ep_rewards

	def _returns_advantages(self, rewards, dones, values, next_value):
		returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

		for t in reversed(range(rewards.shape[0])):
			returns[t] = rewards[t] + self.gamma * returns[t+1] * (1-dones[t])
		returns = returns[:-1]

		advantages = returns - values
		return returns, advantages

	def test(self, env, render=True):
		obs, done, ep_reward = env.reset(), False, 0

		while not done:
			action, _ = self.model.action_value(obs[None, :])
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			if render:
				env.render()
		return ep_reward

	def _value_loss(self, returns, value):
		return self.value_c * kls.mean_squared_error(returns, value)

	def _logits_loss(self, actions_and_advantages, logits):
		actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
		weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

		actions=tf.cast(actions, tf.int32)
		policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

		probs = tf.nn.softmax(logits)
		entropy_loss = kls.categorical_crossentropy(probs, probs)

		return policy_loss - self.entropy_c * entropy_loss




if (__name__ == "__main__"):

	import gym

	env = gym.make('CartPole-v0')
	model = Model(num_actions=env.action_space.n)

	# obs = env.reset()

	# action, value = model.action_value(obs[None, :])
	# print(action, value)

	agent = A2CAgent(model)
	# rewards_sum = agent.test(env)
	# print("%d out of 200" % rewards_sum)

	print("Begin training...")
	rewards_history = agent.train(env)
	print("Finished training, testing...")
	print("%d out of 200" % agent.test(env))

