# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:31:42 2019

@author: MarkZX
"""

# -*- coding: utf-8 -*-

import numpy
import scipy.special
import matplotlib.pyplot
class nueralNetwork:
    def __init__(self,inputnodes,hiddennodes_1,hiddennodes_2,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes_1=hiddennodes_1
        self.hnodes_2=hiddennodes_2
        self.onodes=outputnodes
        self.lr=learningrate
        
        self.wih=numpy.random.normal(0.0,pow(self.hnodes_1,-0.5),(self.hnodes_1,self.inodes))
        self.whh=numpy.random.normal(0.0, pow(self.hnodes_2, -0.5), (self.hnodes_2, self.hnodes_1))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes_2))
        self.activation_function=lambda x:scipy.special.expit(x)
        
        pass
    def train(self, inputs_list, targrts_list):
        inputs = numpy.array(inputs_list,ndmin =2).T
        targets =numpy.array(targrts_list,ndmin =2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs_1=self.activation_function(hidden_inputs)
        
        hidden_inputs_2=numpy.dot(self.whh,hidden_outputs_1)
        hidden_outputs_2=self.activation_function(hidden_inputs_2)
        
        final_inputs=numpy.dot(self.who,hidden_outputs_2)
        final_outputs=self.activation_function(final_inputs)
        
        output_errors=targets-final_outputs
        hidden_errors_1 = numpy.dot(self.who.T,output_errors)
        hidden_errors_2 = numpy.dot(self.whh.T,hidden_errors_1)
        
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs_2))
        self.whh+=self.lr*numpy.dot((hidden_errors_1*hidden_outputs_2*(1.0-hidden_outputs_2)),numpy.transpose(hidden_outputs_1))
        self.wih+=self.lr*numpy.dot((hidden_errors_2*hidden_outputs_1*(1.0-hidden_outputs_1)),numpy.transpose(inputs))
        pass
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list,ndmin =2).T
        hidden_inputs_1=numpy.dot(self.wih,inputs)
        hidden_outputs_1=self.activation_function(hidden_inputs_1)
        
        hidden_inputs_2=numpy.dot(self.whh,hidden_outputs_1)
        hidden_outputs_2=self.activation_function(hidden_inputs_2)
        
        final_inputs=numpy.dot(self.who,hidden_outputs_2)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs
        
    
    pass

input_nodes=784
output_nodes=10
hidden_nodes_1=50
hidden_nodes_2=30
learning_rate=0.2
n= nueralNetwork(input_nodes,hidden_nodes_1,hidden_nodes_2,output_nodes,learning_rate)


#load the test record
test_data_file=open(".\\MNIST_BIG\\mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

#test the network
scorecard=[]
training_data_file=open(".\\MNIST_BIG\\mnist_train.csv",'r')
training_data_list= training_data_file.readlines()
training_data_file.close()

#tain the model
for record in training_data_list:
    all_values = record.split(',')
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    targets= numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])]=0.99
    n.train(inputs,targets)
    pass
for record in test_data_list:
    all_values = record.split(',')
    correct_label=int(all_values[0])    
    outputs=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
    label=numpy.argmax(outputs)
    
    if(label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

#calculate the performance scores,the fraction of correct answers
scorecard_array=numpy.asarray(scorecard)
print("performance =",scorecard_array.sum()/scorecard_array.size)

    


