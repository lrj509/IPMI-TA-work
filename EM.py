import imageio
import numpy as np 
import math 


class EM:

    """

    """

    @staticmethod
    def _GaussianEquasion(value, mean, std):
         first_part = 1/(std * math.sqrt(2*math.pi))
         second_part = math.e ** (-0.5*((value-mean)/std)**2)

         return first_part*second_part


    @staticmethod
    def _Expectation(image, number_of_classes, averages, stds):

        """
        Implimentation of the Expectation-Maximisation algorithm.
        """

        image_shape = np.shape(image)
        image_area = image_shape[0] * image_shape[1]

        image_vector = image.reshape(image_area) #/ 256

        expectation_vector = np.zeros(image_vector.shape[0])

        for key, pixel in enumerate(image_vector):
            
            current_prob = 0
            current_class = None
            for class_num in range(number_of_classes):     
                if EM._GaussianEquasion(pixel, averages[class_num], stds[class_num]) > current_prob:
                    current_class = class_num
                    current_prob = EM._GaussianEquasion(pixel, averages[class_num], stds[class_num])
            expectation_vector[key] = current_class
        

        return image_vector, expectation_vector



    @staticmethod
    def _Maximisation(image_vector, expectation_vector): 

        store_dict = {}

        for key, val in enumerate(expectation_vector):

            if val in store_dict:
                store_dict[val].append(int(image_vector[key]))
            else:
                store_dict[val] = [int(image_vector[key])]

        means = []
        stds = []

        for i in store_dict:
            means.append(sum(store_dict[i])/len(store_dict[i]))
            stds.append(np.std(np.asarray(store_dict[i])))

        return tuple(means), tuple(stds)


    @staticmethod
    def _LoadImage(file_location):

        try:

            image = imageio.imread(file_location)

        except:

            raise IOError("Could not read in image")

        return image


    def __call__(self, file_location, convergence_ratio, no_means = 4, initial_means = (50,100,150,200), initial_stds = (10,10,10,10)):

        image = EM._LoadImage(file_location)

        means, stds = initial_means, initial_stds

        while True: 

            new_means, new_stds = EM._Maximisation(*EM._Expectation(image, no_means, means,stds))
            if all([(means[i]/new_means[i])-1 < convergence_ratio for i in range(len(new_means))]):
                break
            means = new_means
            stds = new_stds
            print(means)

        return means, stds








if __name__ == "__main__":

    m, std = EM()("/home/luke/TAWork/Data/brain_noise.png", 0.001)

   