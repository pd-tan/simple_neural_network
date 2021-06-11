import numpy as np
def generate_data(number_of_data, train_ratio,validate_ratio):
    input_size_ = 32
    number_of_labels_ = 2
    threshold_ = 0
    input_data = np.random.randn(number_of_data,input_size_,1)
    weightage_vector = np.random.randint(low=-4,high=4,size=(input_size_,1))
    weighted_values = input_data*weightage_vector
    summed_values = np.sum(weighted_values,axis=1)
    label_switch = summed_values>threshold_
    print(label_switch)
    truth = np.zeros((number_of_data,number_of_labels_,1))
    # TODO find faster approach to achieve this
    for sample_number in range(number_of_data):
        if label_switch[sample_number,0]:
            truth[sample_number,0,0] = 1
        else:
            truth[sample_number, 1, 0] = 1

    return input_data,truth







if __name__ == '__main__':
    input,truth = generate_data(20,1,1)