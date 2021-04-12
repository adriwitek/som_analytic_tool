from math import ceil
import numpy as np
import networkx as nx


class GSOM:
    def __init__(self, initial_map_size, parent_quantization_error, t1, data_size, weights_map, parent_dataset, neuron_builder):
        assert parent_dataset is not None, "Provided dataset is empty"
        
        self.__neuron_builder = neuron_builder
        self.__data_size = data_size
        self.__t1 = t1
        self.__parent_quantization_error = parent_quantization_error
        self.__initial_map_size = initial_map_size
        self.__parent_dataset = parent_dataset

        self.weights_map = [weights_map]

        self.neurons = self.__build_neurons_list()

    def winner_neuron(self, data):
        number_of_data = 1 if (len(data.shape) == 1) else data.shape[0]
        distances = np.empty(shape=(number_of_data, len(self.neurons.values())))

        neurons_list = list(self.neurons.values())
        for idx, neuron in enumerate(neurons_list):
            distances[:, idx] = neuron.activation(data)

        winner_neuron_per_data = distances.argmin(axis=1)

        support_stuff = [[position for position in np.where(winner_neuron_per_data == neuron_idx)[0]]
                         for neuron_idx in range(len(neurons_list))]

        winner_neurons = [neurons_list[idx] for idx in winner_neuron_per_data]

        return winner_neurons, support_stuff

    def train(self, epochs, initial_gaussian_sigma, initial_learning_rate, decay,
              dataset_percentage, min_dataset_size, seed, maxiter):
        _iter = 0
        can_grow = True
        while can_grow and (_iter < maxiter):
            self.__neurons_training(decay, epochs, initial_learning_rate, initial_gaussian_sigma,
                                    dataset_percentage, min_dataset_size, seed)

            _iter += 1
            print('\t Iteraccion: ', _iter)
            can_grow = self.__can_grow()
            if can_grow:
                self.grow()

        if can_grow:
            self.__map_data_to_neurons()
        return self

    def __neurons_training(self, decay, epochs, learning_rate, sigma, dataset_percentage, min_dataset_size, seed):
        lr = learning_rate
        s = sigma
        for iteration in range(epochs):
            for data in self.__training_data(seed, dataset_percentage, min_dataset_size):
                self.__update_neurons(data, lr, s)

            lr *= decay
            s *= decay

    def __update_neurons(self, data, learning_rate, sigma):
        gauss_kernel = self.__gaussian_kernel(self.winner_neuron(data)[0][0], sigma)

        for neuron in self.neurons.values():
            weight = neuron.weight_vector()
            weight += learning_rate * gauss_kernel[neuron.position] * (data - weight)
            self.weights_map[0][neuron.position] = weight

    def __gaussian_kernel(self, winner_neuron, gaussian_sigma):
        # computing gaussian kernel
        winner_row, winner_col = winner_neuron.position
        s = 2 * (gaussian_sigma ** 2)

        gauss_col = np.power(np.arange(self.map_shape()[1]) - winner_col, 2) / s
        gauss_row = np.power(np.arange(self.map_shape()[0]) - winner_row, 2) / s

        return np.outer(np.exp(-1 * gauss_row), np.exp(-1 * gauss_col))

    def __can_grow(self):
        self.__map_data_to_neurons()

        MQE = 0.0
        mapped_neurons = 0
        changed_neurons = 0

        #anniadido
        assert self.__parent_quantization_error is not None, "Parent Quantization Error must not be None"

        for neuron in self.neurons.values():
            changed_neurons += 1 if neuron.has_changed_from_previous_epoch() else 0
            if neuron.has_dataset():
                MQE += neuron.compute_quantization_error()
                mapped_neurons += 1

        #TODO
        #print('condicion',((MQE / mapped_neurons) >= (self.__t1 * self.__parent_quantization_error)), str((MQE / mapped_neurons)) )
        return ((MQE / mapped_neurons) >= (self.__t1 * self.__parent_quantization_error)) 
        #Quitada esta clausula
        # and \ (changed_neurons > int(np.round(mapped_neurons/5)))

    def __map_data_to_neurons(self):
        self.__clear_neurons_dataset()

        # finding the new association for each neuron
        _, support_stuff = self.winner_neuron(self.__parent_dataset)

        neurons = list(self.neurons.values())
        for idx, data_idxs in enumerate(support_stuff):
            neurons[idx].replace_dataset(self.__parent_dataset[data_idxs, :])

    def __clear_neurons_dataset(self):
        for neuron in self.neurons.values():
            neuron.clear_dataset()

    def __find_error_neuron(self,):
        # self.__map_data_to_neurons()

        quantization_errors = list()
        for neuron in self.neurons.values():
            quantization_error = -np.inf
            if neuron.has_dataset():
                quantization_error = neuron.compute_quantization_error()
            quantization_errors.append(quantization_error)

        idx = np.unravel_index(np.argmax(quantization_errors), dims=self.map_shape())
        return self.neurons[idx]

    def __find_most_dissimilar_neuron(self, error_neuron):
        weight_distances = dict()
        for neuron in self.neurons.values():
            if self.are_neurons_neighbours(error_neuron, neuron):
                weight_distances[neuron] = error_neuron.weight_distance_from_other_unit(neuron)

        return max(weight_distances, key=weight_distances.get)

    def grow(self):
        error_neuron = self.__find_error_neuron()
        dissimilar_neuron = self.__find_most_dissimilar_neuron(error_neuron)

        if self.are_in_same_row(error_neuron, dissimilar_neuron):
            new_neuron_idxs = self.add_column_between(error_neuron, dissimilar_neuron)
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "horizontal")
        elif self.are_in_same_column(error_neuron, dissimilar_neuron):
            new_neuron_idxs = self.add_row_between(error_neuron, dissimilar_neuron)
            self.__init_new_neurons_weight_vector(new_neuron_idxs, "vertical")
        else:
            raise RuntimeError("Error neuron and the most dissimilar are not adjacent")

    def add_column_between(self, error_neuron, dissimilar_neuron):
        error_col = error_neuron.position[1]
        dissimilar_col = dissimilar_neuron.position[1]
        new_column_idx = max(error_col, dissimilar_col)

        map_rows, map_cols = self.map_shape()

        new_line_idx = [(row, new_column_idx) for row in range(map_rows)]

        for row in range(map_rows):
            for col in reversed(range(new_column_idx, map_cols)):
                new_idx = (row, col + 1)
                neuron = self.neurons.pop((row, col))
                neuron.position = new_idx
                self.neurons[new_idx] = neuron

        line = np.zeros(shape=(map_rows, self.__data_size), dtype=np.float32)
        self.weights_map[0] = np.insert(self.weights_map[0], new_column_idx, line, axis=1)

        return new_line_idx

    def add_row_between(self, error_neuron, dissimilar_neuron):
        error_row = error_neuron.position[0]
        dissimilar_row = dissimilar_neuron.position[0]
        new_row_idx = max(error_row, dissimilar_row)

        map_rows, map_cols = self.map_shape()

        new_line_idx = [(new_row_idx, col) for col in range(map_cols)]

        for row in reversed(range(new_row_idx, map_rows)):
            for col in range(map_cols):
                new_idx = (row + 1, col)
                neuron = self.neurons.pop((row, col))
                neuron.position = new_idx
                self.neurons[new_idx] = neuron

        line = np.zeros(shape=(map_cols, self.__data_size), dtype=np.float32)
        self.weights_map[0] = np.insert(self.weights_map[0], new_row_idx, line, axis=0)

        return new_line_idx

    def __init_new_neurons_weight_vector(self, new_neuron_idxs, new_line_direction):
        for row, col in new_neuron_idxs:
            adjacent_neuron_idxs = self.__get_adjacent_neuron_idxs_by_direction(row, col, new_line_direction)
            weight_vector = self.__mean_weight_vector(adjacent_neuron_idxs)

            self.weights_map[0][row, col] = weight_vector
            self.neurons[(row, col)] = self.__build_neuron((row, col))

    def __mean_weight_vector(self, neuron_idxs):
        weight_vector = np.zeros(shape=self.__data_size, dtype=np.float32)
        for adjacent_idx in neuron_idxs:
            weight_vector += 0.5 * self.neurons[adjacent_idx].weight_vector()
        return weight_vector

    @staticmethod
    def __get_adjacent_neuron_idxs_by_direction(row, col, direction):
        adjacent_neuron_idxs = list()
        if direction == "horizontal":
            adjacent_neuron_idxs = [(row, col - 1), (row, col + 1)]

        elif direction == "vertical":
            adjacent_neuron_idxs = [(row - 1, col), (row + 1, col)]

        return adjacent_neuron_idxs

    @staticmethod
    def are_neurons_neighbours(first_neuron, second_neuron):
        return np.linalg.norm(np.asarray(first_neuron.position) - np.asarray(second_neuron.position), ord=1) == 1

    @staticmethod
    def are_in_same_row(first_neuron, second_neuron):
        return abs(first_neuron.position[0] - second_neuron.position[0]) == 0

    @staticmethod
    def are_in_same_column(first_neuron, second_neuron):
        return abs(first_neuron.position[1] - second_neuron.position[1]) == 0

    def __build_neurons_list(self):
        rows, cols = self.__initial_map_size
        return {(x, y): self.__build_neuron((x, y)) for x in range(rows) for y in range(cols)}

    def __build_neuron(self, weight_position):
        return self.__neuron_builder.new_neuron(self.weights_map, weight_position)

    def map_shape(self):
        shape = self.weights_map[0].shape
        return shape[0], shape[1]


    def __training_data(self, seed, dataset_percentage, min_size):
        dataset_size = len(self.__parent_dataset)
        if dataset_size <= min_size:
            iterator = range(dataset_size)
        else:
            iterator = range(int(ceil(dataset_size * dataset_percentage)))

        random_generator = np.random.RandomState(seed)
        for _ in iterator:
            yield self.__parent_dataset[random_generator.randint(dataset_size)]



    ###############################################################
    ###                 added by adriwitek                      ###
    ###############################################################

    def get_weights_map(self):
        return self.weights_map[0]

    def get_neurons(self):
        return self.neurons


    def get_structure_graph(self,grafo,dataset, level=0):
       


        #Indexes of dataset mapped on each neuron
        mapping = [[list() for _ in range(self.map_shape()[1])] for _ in range(self.map_shape()[0])]
        # contains targets list per neuron, later used to analyze the graph
        neurons_mapped_targets = {} 
        data = dataset[:,:-1]

        # Getting winnig neurons for each data element
        for i,d in enumerate(data):
            winner_neuron = self.winner_neuron(d)[0][0]
            r, c = winner_neuron.position
            mapping[r][c].append(i)

            if((r,c) in neurons_mapped_targets):
                neurons_mapped_targets[(r,c)].append(dataset[i][-1]) 
            else:
                neurons_mapped_targets[(r,c)] = []
                neurons_mapped_targets[(r,c)].append(dataset[i][-1]) 

        grafo.add_node(self,nivel =level, neurons_mapped_targets = neurons_mapped_targets )
 

        #Creating the graph itself
        for neuron in self.neurons.values():
            if neuron.child_map is not None:

                grafo.add_node(neuron.child_map ,
                                nivel=level+1, 
                                neurona_padre_pos= neuron.position, 
                                )
                grafo.add_edge(self,neuron.child_map)
            
                r, c = neuron.position
                index_list= mapping[r][c]
                grafo = neuron.child_map.get_structure_graph(grafo,dataset[index_list], level=level+1)


        return grafo


    def get_map_qe_and_mqe(self):

        self.__map_data_to_neurons()

        MAP_QE = 0.0
        mapped_neurons = 0

        for neuron in self.neurons.values():
            if neuron.has_dataset():
                MAP_QE += neuron.compute_quantization_error()
                mapped_neurons += 1

        return MAP_QE , (MAP_QE / mapped_neurons)