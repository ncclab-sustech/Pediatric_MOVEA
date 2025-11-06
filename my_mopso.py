#######################################################
# Autora: Elena Romero Contreras
# Código basado en http://delta.cs.cinvestav.mx/~ccoello/EMOO/MOPSO.py
# Clase partícula e implementación del algoritmo MOPSO
# y sus funciones auxiliares
#######################################################

import random
import numpy as np
from public.util import *
from public import plot


class my_mopso:

	def __init__(self, pop_size, pop_div, M):
		self.pop_size = pop_size
		self.pop_div = pop_div
		self.M = M
		self.min = -1
		self.max = 1
		self.p = np.random.uniform(self.min, self.max, (pop_size, pop_div))
		self.v = np.zeros((pop_size, pop_div))
		self.fit_temp = np.zeros((pop_size, M))
		self.archive = self.p
		self.archive_size = 30
		self.pbest = self.p
		self.plot_ = plot.Plot_pareto()

	def evaluate(self, population):
		self.fit_temp = np.zeros((len(population), self.M))
		self.fit_temp[:, 0] = np.array([function1_s(s) for s in population])
		self.fit_temp[:, 1] = np.array([function2_s(s) for s in population])
		return self.fit_temp

	def pareto(self):  # 可以只计算font1

		population = np.vstack([self.p, self.archive])
		size = 0
		self.evaluate(population)
		# fit_temp = np.zeros((len(population), self.M))
		# fit_temp[:, 0] = np.array([function1_s(s) for s in population])
		# fit_temp[:, 1] = np.array([function2_s(s) for s in population])
		cv_value = np.array([h(s) for s in population])
		font = fast_non_dominated_sort(self.fit_temp[:, 0], self.fit_temp[:, 1], cv_value)###
		if len(font[0]) <= self.archive_size:
			self.archive = population[font[0]]
		else:
			crowding_distance_values = crowding_distance(self.fit_temp[:, 0], self.fit_temp[:, 1], font[0])
			sort = sort_by_values([x for x in range(len(crowding_distance_values))], crowding_distance_values)  # [4,2,1,3]
			front2 = [font[0][sort[j]] for j in range(len(font[0]))]
			front2.reverse()  # 该PF按拥挤度排序
			for value in front2:
				if self.archive_size > size:
					size = size + 1
					self.archive = np.vstack([self.archive, population[value]])
				else:
					break

	def updateVelocity(self, w, c1=2, c2=2 ):  # w = 0.9 - ((0.9 - 0.4)/max_iter)*i

		for i in range(self.pop_size):
			leader = self.archive[np.random.choice(len(self.archive))]
			for j in range(self.pop_div):
				r1 = random.random()
				r2 = random.random()
				vel_cognitive = c1 * r1 * (self.pbest[i][j] - self.p[i][j])
				vel_social = c2 * r2 * (leader[j] - self.p[i][j])
				self.v[i][j] = w * self.v[i][j] + vel_cognitive + vel_social

	def updatePosition(self):
		for i in range(self.pop_size):
			for j in range(self.pop_div):
				self.p[i][j] = self.p[i][j] + self.p[i][j]
			if self.checkCV(self.p[i],self.pbest[i]):
				self.pbest[i] = self.p[i]

	def checkCV(self, x, y):  # x domain y

		if h(x) == h(y) == 0:
			if (function1_s(x) <= function1_s(y) and function1_s(x) < function1_s(y)) \
					or (function1_s(x) < function1_s(y) and function1_s(x) <= function1_s(y)):
				return True
		elif h(x) < h(y):
			return True
		else:
			return False

	def checkLimits(self):
		for i in range(self.pop_size):
			for j in range(self.pop_div):
				if self.p[i][j] > 1:
					self.p[i][j] = 1
				elif self.p[i][j] < -1:
					self.p[i][j] = -1

	def train(self, maxgen):


		for gen in range(maxgen):
			self.pareto()
			self.updateVelocity(0.9 - ((0.9 - 0.4)/maxgen)*gen)
			self.updatePosition()
			self.checkLimits()
			print("p_shape", self.p.shape)
			#print("eva_shape", self.evaluate(self.p).shape)
			if (gen+1) % 10 == 0:
					h(self.p[0])
					print(self.p[0])
				#self.plot_.show(self.p, self.evaluate(self.p), self.archive, self.evaluate(self.archive), i, self.M)


if __name__ == "__main__":

	mopso = my_mopso(100, 125, 2)
	mopso.train(100)
