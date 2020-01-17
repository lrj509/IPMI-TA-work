import imageio
import numpy as np 
import math 


class EM:

    """
    This class contains various methods relating to the 
    Expectation-Maximisation (EM) algorithm. 

    @author Luke Jenkinson
    """



    @staticmethod
    def _GaussianEquasion(value, mean, std):

        """
        This method calculates the probablity density value
        at a given value, with a known mean and known standard
        deviation. This is an implimentation of the Equasion 
        found: https://en.wikipedia.org/wiki/Normal_distribution

        @param value:   
                Query value

        @param mean:    
                The mean of the distribution

        @param std:     
                The standard deviation of the distribution


        @returns        
                The PDF value at the query value. 
        """

        first_part = 1/(std * math.sqrt(2*math.pi))
        second_part = math.e ** (-0.5*((value-mean)/std)**2)

        return first_part*second_part


    @staticmethod
    def _Expectation(image, number_of_classes, averages, stds):

        """
        Implimentation of the Expectation section of the EM
        algorithm. This method calculates which classes the 
        image pixels most likely belong to, given the current 
        means (averages) and their relevent standard deviations. 

        @param image:   
                A numpy array containing the image pixel values
                as a 2D array. 

        @param number_of_classes:  
                The number of classes to segment the image into. 

        @param averages: 
                A tuple containing the current means

        @param stds:    
                A tuple containing the current standard deviations. 


        @returns
                A numpy vector containing the current most likely 
                pixel classes for each pixel. 
        """

        image_vector = image.reshape(image.shape[0]*image.shape[1])

        expectation_vector = np.zeros(image_vector.shape[0])

        for key, pixel in enumerate(image_vector):
            
            current_prob = 0
            current_class = None

            for class_num in range(number_of_classes):    

                query_pdf = EM._GaussianEquasion(pixel, 
                    averages[class_num], 
                    stds[class_num]) 

                if query_pdf > current_prob:

                    current_class = class_num
                    current_prob = query_pdf

            expectation_vector[key] = current_class     

        return image_vector, expectation_vector



    @staticmethod
    def _Maximisation(image_vector, expectation_vector):

        """
        The maximisation part of the algorithm. This method
        takes the image vector and the currently assinged 
        classes, and then returns an updated tuple of means
        and an updated tuple of stadard deviations. 

        @param image_vector:  
                A numpy array containing the image pixel values 
                in vector form. 

        @param expectation_vector: 
                A numpy vector containing the current most 
                likely pixel classes for each pixel. 


        @returns 
                The updated means and updated standard deviations
        """ 

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

        """
        Loads in the image with basic error handling. 

        @param file_locataion: 
                The full or relative path of the file location.

        @returns
                The image as a 2D numpy array.  
        """

        try:

            image = imageio.imread(file_location)

        except:

            raise IOError("Could not read in image")

        return image


    def _CalcPriors(priors): 

        """
        Given a tuple of paths to files, this method calculates 
        the bayesian priors. 

        @param priors:  
                Tuple containing paths to prior files.  

        @returns 
                The prior means and standard deviations. 
        """

        pass

    def __call__(self, 
        file_location, 
        convergence_ratio, 
        no_means = 4, 
        priors = None, 
        initial_means = (50,100,150,200), 
        initial_stds = (10,10,10,10)):

        """
        Overloading the __call__ method, so that an instantiated
        class can be called to run the EM algorithm. 

        @param file_location: 
                The file location of the image to be segmented. 
                Can be full or relative. 

        @param convergence_ratio: 
                The difference between sucsessive runs of the 
                EM algorithm.

        @param no_means:       
                The number of classes to segment into. 

        @param priors:
                Tuple of file locations of the prior files. None 
                if there are no priors. 

        @param initial_means:  
                Starting point for the algorithm

        @param initial_stds:   
                The starting standard deviations


        @returns 
            The final calculated means and standard deviations. 

        """

        image = EM._LoadImage(file_location)

        if priors is not None: 

            means, stds = _CalcPriors(priors)

        else:
            means, stds = initial_means, initial_stds

        while True: 

            expectation = EM._Expectation(image, no_means, means,stds)
            new_means, new_stds = EM._Maximisation(*expectation)
            
            if all([(means[i]/new_means[i])-1 < convergence_ratio for i in range(len(new_means))]):
                break
            means = new_means
            stds = new_stds
            print(means)

        return means, stds


if __name__ == "__main__":

    m, std = EM()("/home/luke/TAWork/IPMI-TA-work/Data/brain.png", 0.001)

   