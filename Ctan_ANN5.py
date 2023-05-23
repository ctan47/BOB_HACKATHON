import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))
        
    def train(self, input_patterns, output_patterns):
        for input_pattern, output_pattern in zip(input_patterns, output_patterns):
            self.weights += np.outer(input_pattern, output_pattern)
    
    def recall(self, input_pattern):
        output_pattern = np.sign(np.dot(input_pattern, self.weights.T))
        return output_pattern

# Example usage:
input_patterns = np.array([[1, -1, 1, -1],
                           [1, 1, -1, -1]])
output_patterns = np.array([[1, -1],
                            [-1, 1]])

bam = BAM(input_size=4, output_size=2)
bam.train(input_patterns, output_patterns)

input_pattern = np.array([1, -1, 1, -1])
output_pattern = bam.recall(input_pattern)
print("Input Pattern:", input_pattern)
print("Output Pattern:", output_pattern)
