# -*- coding: utf-8 -*-
"""
BASELINE Q-Learning Model + SIDE BIAS
Parameters: alpha (fixed learning rate), beta (inverse temperature), kappa (side bias)
NO stimulation, NO lapse, NO kernel, NO decay, NO adaptive learning

Identical to TwoStageQLearning_Baseline except for one term in the softmax:
	logit P(choose target 2) = beta * (Q_high - Q_low) + kappa * s
where s = +1 when targetH is on the right (targetH[:,2] == 0) and -1 when on the left (== 1).
kappa > 0 is a rightward pull, in logit units. It sits outside beta, so it does not
trade off against beta. The Q update is untouched: side moves the choice, not what is learned.
"""

import numpy as np


def Calc_DistrQlearning_2Targs_BaselineSide(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice, targetH_side):
	'''Baseline Q-learning with a side bias. targetH_side: 1 = left, 0 = right, one per trial'''
	alpha = parameters[0]  # Fixed learning rate
	beta = parameters[1]   # Inverse temperature
	kappa = parameters[2]  # Side bias (+ = rightward)

	s = 1.0 - 2.0 * np.asarray(targetH_side, dtype=float)  # +1 when targetH is on the right (side code 0)

	Q_low = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))
	Q_low[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))
	prob_choice_low[0] = 0.5
	prob_choice_high[0] = 0.5

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# STANDARD Q-LEARNING UPDATE (no modifications)
		if chosen_target[i] == 1:  # LV chosen
			delta = float(rewards[i]) - Q_low[i]
			Q_low[i+1] = Q_low[i] + alpha * delta
			Q_high[i+1] = Q_high[i]  # Unchosen option unchanged
		elif chosen_target[i] == 2:  # HV chosen
			delta = float(rewards[i]) - Q_high[i]
			Q_high[i+1] = Q_high[i] + alpha * delta
			Q_low[i+1] = Q_low[i]  # Unchosen option unchanged

		# Softmax decision rule with side bias (free-choice only)
		if instructed_or_freechoice[i+1] == 2:
			prob_choice_low[i+1] = 1./(1. + np.exp(beta*(Q_high[i+1] - Q_low[i+1]) + kappa*s[i+1]))
			prob_choice_high[i+1] = 1. - prob_choice_low[i+1]

			accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or
			                               (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))
			log_prob_total += np.log(prob_choice_low[i+1]*(chosen_target[i+1]==1) +
			                         prob_choice_high[i+1]*(chosen_target[i+1]==2))
		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total


def loglikelihood_DistrQlearning_2Targs_BaselineSide(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice, targetH_side):
	Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total = \
		Calc_DistrQlearning_2Targs_BaselineSide(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice, targetH_side)
	return log_prob_total
