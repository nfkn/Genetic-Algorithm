# -*-coding:utf-8 -*-

import math
import numpy as np
import random
from matplotlib import pyplot as plt


class GA_int_encoding:

    # 函数功能：GA类的构造函数
    # 函数输入：
    # 函数输出：
    def __init__(self, population_size, crossover_probability, mutation_probability, coordinate):
        self.coordinate = coordinate
        self.population_size = population_size                                      # 种群数量
        self.phenotype_size = self.coordinate.shape[0]                              # 表现型的数量
        self.phenotype_max = self.phenotype_size - 1                                # 表现型的最大取值
        self.fitness = np.zeros((self.population_size, 3))                          # 第一列为总适应度，第二列为选择概率，第三列为累计概率
        self.crossover_probability = crossover_probability                          # 染色体交叉的概率
        self.mutation_probability = mutation_probability                            # 染色体变异的概率
        self.best_fit = np.zeros((1, self.phenotype_size))                          # 种群中最优个体
        self.best_length = 0                                                        # 种群中最优个体的取值
        self.population = self.species_origin()                                     # 种群，整数编码时population中存放的就是表现型
        self.dist_mat = self.get_dist_mat()                                         # 距离矩阵，i到j的矩阵
        print("begin GA")

    # 函数功能：计算两两城市之间的距离
    # 函数输入：coordinates城市坐标，num_city城市数量
    # 函数输出：dist_mat距离矩阵
    def get_dist_mat(self):
        dist_mat = np.zeros((self.phenotype_size, self.phenotype_size))
        for i in range(self.phenotype_size):
            for j in range(i, self.phenotype_size):
                dist_mat[i][j] = dist_mat[j][i] = np.linalg.norm(self.coordinate[i] - self.coordinate[j])
        return dist_mat

    # 函数功能：初始化生成population_size(种群数量)，phenotype_size(表现型数量)的种群，一行代表一个种群
    # 函数输入：population_size（行数）,phenotype_size（列数）
    # 函数输出：返回一个population_size行，phenotype_size列的二维numpy对象
    def species_origin(self):
        population = np.zeros((self.population_size, self.phenotype_size), dtype=int)
        for individual in range(self.population_size):
            population[individual] = np.random.permutation(range(0, self.phenotype_size))
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
                distance += self.dist_mat[count1][count2]
            # 回到起始点
            count1 = int(self.population[individual][0])
            count2 = int(self.population[individual][phenotype])
            distance += self.dist_mat[count1][count2]
            self.fitness[individual][0] = 1/distance
            # ----------------------------------------------------------------------- #
        # 计算适应度
        self.fitness[..., -3] = np.sum(self.fitness, axis=1).T
        # 计算选择概率
        self.fitness[..., -2] = self.fitness[..., -3] / np.sum(self.fitness[..., -3])
        # 计算累计概率
        self.fitness[..., -1] = np.cumsum(self.fitness[..., -2]).T
        self.best_fit = self.population[np.argmax(self.fitness[..., -3]), ...]
        self.best_length = 1/np.max(self.fitness[..., -3])
        # print('fitness is:', self.fitness)

    # 函数功能：按照轮盘赌的方式选择个体
    # 函数输入：适应度的累计概率
    # 函数输出：选择完成后的新种群
    def selection(self):
        new_population = np.zeros((self.population_size, self.phenotype_size), dtype=int)
        temp_fitness = np.zeros((self.population_size, 1))
        # 生成轮盘概率
        wheel_probability = np.random.rand(self.population_size, 1)
        for individual_count in range(self.population_size):
            temp_fitness = self.fitness[..., -1] - wheel_probability[individual_count]
            new_population[individual_count, ...] = self.population[np.where(temp_fitness.T > 0)[0][0], ...]
        self.population = new_population
        self.population_size, self.phenotype_size = np.shape(self.population)
        # print('new population is:', self.population)

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
            # ------------------------------------------- #
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
        for count in range(200000):
            self.fitness_function()
            self.selection()
            print(self.best_length)
            if count == 199999:
                print(self.best_fit)
                break
            self.crossover()
            self.mutation()


if __name__ == '__main__':

    coordinates = np.array([
        [671, 236],
        [196, 567],
        [256, 186],
        [221, 617],
        [512, 517],
        [284, 232],
        [557, 468],
        [541, 50],
        [123, 126],
        [575, 359],
        [265, 276],
        [606, 63],
        [483, 572],
        [242, 90],
        [297, 703],
        [406, 769],
        [260, 602],
        [38, 98],
        [524, 554],
        [503, 540],
        [343, 291],
        [129, 426],
        [229, 113],
        [356, 283],
        [303, 151],
        [203, 695],
        [258, 689],
        [678, 750],
        [128, 753],
        [4, 193],
        [2, 431],
        [731, 501],
        [392, 101],
        [689, 319],
        [712, 428],
        [141, 662],
        [88, 530],
        [641, 649],
        [17, 643],
        [613, 746],
        [725, 634],
        [779, 78],
        [751, 46],
        [756, 23],
        [486, 22],
        [671, 415],
        [234, 543],
        [242, 366],
        [154, 416],
        [653, 571],
        [312, 392],
        [717, 348],
        [287, 162],
        [8, 216],
        [481, 183],
        [649, 188],
        [642, 572],
        [724, 611],
        [114, 232],
        [792, 388],
        [82, 473],
        [579, 116],
        [607, 596],
        [772, 626],
        [204, 138],
        [208, 767],
        [253, 456],
        [538, 341],
        [66, 476],
        [385, 604],
        [104, 766],
        [742, 487],
        [386, 60],
        [226, 20],
        [280, 359],
        [279, 799],
        [158, 183],
        [517, 122],
        [606, 181],
        [641, 406],
        [639, 334],
        [393, 79],
        [210, 679],
        [358, 178],
        [603, 615],
        [174, 252],
        [343, 588],
        [93, 482],
        [113, 2],
        [607, 93],
        [84, 475],
        [747, 418],
        [159, 290],
        [185, 142],
        [189, 639],
        [505, 262],
        [177, 293],
        [283, 72],
        [305, 337],
        [635, 109]
    ])
    # x = [0, 1, 3.7, 3, 4, 2.3, 4, 5.5, 4, 3, 2, 1, 0, 10, 0, 9.6, 0, 1, 3.7, 3, 4, 2.3, 4, 5.5, 4, 3]
    # y = [0, 8.0, 0, 4, 0, 1, 3.6, 3, 7.3, 4, 9.6, 4, 8.6, 6.1, 2, 1, 4, 5.5, 4, 3, 2, 1, 0, 10, 0, 9.6]
    x = list(coordinates[:, 0])
    y = list(coordinates[:, 1])

    GA = GA_int_encoding(1000, 0.001, 0.001, coordinates)
    GA.GA_progress()
    pox, poy = [], []
    for index in list(GA.best_fit):
        pox.append(x[index])
        poy.append(y[index])
    pox.append(x[list(GA.best_fit)[0]])
    poy.append(y[list(GA.best_fit)[0]])
    plt.plot(pox, poy, 'g-x')
    plt.show()
