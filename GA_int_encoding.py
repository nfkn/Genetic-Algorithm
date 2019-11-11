# -*-coding:utf-8 -*-

import math
import numpy as np
import random
from matplotlib import pyplot as plt


class GA_int_encoding:

    # 函数功能：GA类的构造函数
    # 函数输入：
    # 函数输出：
    def __init__(self,  population_size, crossover_probability, mutation_probability, x, y):
        self.x = x
        self.y = y
        self.population_size = population_size                                      # 种群数量
        self.phenotype_size = len(self.x)                                           # 表现型的数量
        self.phenotype_max = self.phenotype_size - 1                                # 表现型的最大取值
        self.fitness = np.zeros((self.population_size, 3))                          # 第一列为总适应度，第二列为选择概率，第三列为累计概率
        self.crossover_probability = crossover_probability                          # 染色体交叉的概率
        self.mutation_probability = mutation_probability                            # 染色体变异的概率
        self.best_fit = []                          # 种群中最优个体
        self.best_length = []                                                        # 种群中最优个体的取值
        self.population = self.species_origin()                                     # 种群，整数编码时population中存放的就是表现型
        print("begin GA")

    # 函数功能：初始化生成population_size(种群数量)，phenotype_size(表现型数量)的种群，一行代表一个种群
    # 函数输入：population_size（行数）,phenotype_size（列数）
    # 函数输出：返回一个population_size行，phenotype_size列的二维numpy对象
    def species_origin(self):
        population = np.zeros((self.population_size, self.phenotype_size), dtype=int)
        for individual in range(self.population_size):
            population[individual] = np.array(random.sample(range(0, self.phenotype_size), self.phenotype_size))
        return population

    # 函数功能：为种群中的每个表现型计算适应度，并为每个个体计算总的适应度
    # 函数输入：种群的phenotype表现型，population中存放的就是表现型
    # 函数输出：每个个体表现型所展现出来的fitness适应度，
    #          倒数第三列为每个个体的所有表现型适应度和(总适应度)，
    #          倒数第二列为适应度总占比(选择概率)
    #          倒数第一列为累计概率
    def fitness_function(self):
        self.fitness = np.zeros((self.population_size, 3))  # 重置一下适应度
        individuals, phenotypes = self.population.shape

        for individual in range(individuals):
            # 按照每个表现型计算所展现出来的fitness适应度,依照实际情况设计
            # ----------------------------------------------------------------------- #
            distance = 0.0
            for phenotype in range(1, phenotypes):
                count1 = int(self.population[individual][phenotype])
                count2 = int(self.population[individual][phenotype - 1])
                distance += ((self.x[count1] - self.x[count2])**2 + (self.y[count1] - self.y[count2])**2)**0.5
            # 回到起始点
            count1 = int(self.population[individual][0])
            count2 = int(self.population[individual][phenotype])
            distance += ((self.x[count1] - self.x[count2]) ** 2 + (self.y[count1] - self.y[count2]) ** 2) ** 0.5
            self.fitness[individual][0] = 1/distance
            # ----------------------------------------------------------------------- #
        # 计算适应度
        self.fitness[..., -3] = np.sum(self.fitness, axis=1).T
        # 计算选择概率
        self.fitness[..., -2] = self.fitness[..., -3] / np.sum(self.fitness[..., -3])
        # 计算累计概率
        self.fitness[..., -1] = np.cumsum(self.fitness[..., -2]).T
        self.best_fit.append(self.population[np.argmax(self.fitness[..., -3]), ...])
        self.best_length.append(1/np.max(self.fitness[..., -3]))

    # 函数功能：按照轮盘赌的方式选择个体
    # 函数输入：适应度的累计概率
    # 函数输出：选择完成后的新种群
    def selection(self):
        new_population = np.zeros((self.population_size, self.phenotype_size), dtype=int)
        individual_count = 0
        fitness_count = 0
        # 生成轮盘概率
        wheel_probability = np.random.rand(self.population_size, 1)
        while individual_count < self.population_size:
            if wheel_probability[individual_count] > self.fitness[fitness_count][-1]:
                fitness_count += 1
            else:
                new_population[individual_count, ...] = self.population[fitness_count, ...]
                fitness_count = 0
                individual_count += 1
        self.population = new_population
        self.population_size, self.phenotype_size = np.shape(self.population)

    # 函数功能：个体染色体交叉
    # 函数输入：种群基因
    # 函数输出：染色体交叉完成后的新种群
    def crossover(self):
        # 根据交叉的概率确定交叉的次数
        for count in range(int(self.crossover_probability * self.population_size * self.phenotype_size)):
            gene_begin = np.random.randint(0, self.phenotype_size-1)
            gene_end = np.random.randint(gene_begin, self.phenotype_size)
            individual1 = np.random.randint(0, self.population_size)
            individual2 = np.random.randint(0, self.population_size)
            # 交叉，并保证交叉之后的个体合法
            temp2, temp1 = np.zeros((2, gene_end - gene_begin), dtype=int)
            temp1 = np.copy(self.population[individual1][gene_begin:gene_end])
            temp2 = np.copy(self.population[individual2][gene_begin:gene_end])
            # 交叉
            self.population[individual1][gene_begin:gene_end] = temp2
            self.population[individual2][gene_begin:gene_end] = temp1
            temp1, temp2 = [], []
            # 映射保证合法性
            individual_range = list(range(0, gene_begin)) + list(range(gene_end, self.phenotype_size))
            # 查在temp1和temp2中都有的数字，并删除
            for index_temp in range(len(self.population[individual1][gene_begin:gene_end])):
                if self.population[individual1][gene_begin + index_temp] not in self.population[individual2][gene_begin:gene_end]:
                    temp1.append(self.population[individual1][gene_begin + index_temp])
                if self.population[individual2][gene_begin + index_temp] not in self.population[individual1][gene_begin:gene_end]:
                    temp2.append(self.population[individual2][gene_begin + index_temp])
            for index_temp in range(len(temp1)):
                for index_individual in individual_range:
                    if self.population[individual1][index_individual] == temp1[index_temp]:
                        self.population[individual1][index_individual] = temp2[index_temp]
                    if self.population[individual2][index_individual] == temp2[index_temp]:
                        self.population[individual2][index_individual] = temp1[index_temp]
            # ------------------------------------------- #

    # 函数功能：个体染色体突变
    # 函数输入：种群基因
    # 函数输出：染色体突变完成后的新种群
    def mutation(self):
        for count in range(int(self.mutation_probability * self.population_size * self.phenotype_size)):
            individual = np.random.randint(0, self.population_size)
            phenotype_p1 = np.random.randint(0, self.phenotype_size)
            phenotype_p2 = np.random.randint(0, self.phenotype_size)
            self.population[individual][phenotype_p1], self.population[individual][phenotype_p2] = \
                self.population[individual][phenotype_p2], self.population[individual][phenotype_p1]

    # 函数功能：遗传算法的过程
    def GA_progress(self):
        for count in range(400):
            self.fitness_function()
            self.selection()
            print(self.best_length[-1])
            print(count)
            if count == 399:

                # print(self.best_fit)
                break
            self.crossover()
            self.mutation()


if __name__ == '__main__':
    x = [0, 1, 3.7, 3, 4, 2.3, 4, 5.5, 4, 3, 2, 1, 0, 10, 0, 9.6]
    y = [0, 8.0, 0, 4, 0, 1, 3.6, 3, 7.3, 4, 9.6, 4, 8.6, 6.1, 2, 1]

    GA = GA_int_encoding(250, 0.01, 0.01, x, y)
    GA.GA_progress()
    pox, poy = [], []
    min_position = GA.best_length.index(min(GA.best_length))
    for index in list(GA.best_fit[min_position]):
        pox.append(x[index])
        poy.append(y[index])
    pox.append(x[list(GA.best_fit[min_position])[0]])
    poy.append(y[list(GA.best_fit[min_position])[0]])
    plt.figure(1)
    plt.plot(pox, poy, 'g-x')
    plt.figure(2)
    plt.plot(GA.best_length, 'g-')
    plt.show()
