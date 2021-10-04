from sys import path
import matplotlib.pyplot as plt
import numpy as np
FILE_TO_OPEN = "gym_3.log"

file = open(FILE_TO_OPEN, 'r')
line:str = file.readline()
coff1_data = {'0': [0]}
coff2_data = {'0': [0]}
coff3_data = {'0': [0]}
weightsPlaceHolder = (0,0,0)
while(line):
    if(line[0] == 'o'):
        tempSplit = line.split(":")
        tempSplit = tempSplit[1].replace("(","").replace(")","").split(',')
        weightsPlaceHolder = (tempSplit[0],tempSplit[1],tempSplit[2])
    if(len(line) > 1 and line[1] == 'w'):
        tempSplit = line.split(":")
        if(weightsPlaceHolder[0] in coff1_data):
            oldData = coff1_data.get(weightsPlaceHolder[0])
            coff1_data[float(weightsPlaceHolder[0])].append(float(tempSplit[1]))
        else:
            coff1_data[float(weightsPlaceHolder[0])] = [float(tempSplit[1])]

        if(weightsPlaceHolder[1] in coff2_data):
            coff2_data[float(weightsPlaceHolder[1])].append(float(tempSplit[1]))
        else:
            coff2_data[float(weightsPlaceHolder[1])] = [float(tempSplit[1])]

        if(weightsPlaceHolder[2] in coff3_data):
            coff3_data[float(weightsPlaceHolder[2])].append(float(tempSplit[1]))
        else:
            coff3_data[float(weightsPlaceHolder[2])] = [float(tempSplit[1])]
    line = file.readline()

for key in coff1_data:
    data = coff1_data[key]
    if(not key == "0"):
        avg = sum(data) / len(data)
        coff1_data[key].clear()
        coff1_data[key] = avg

for key in coff2_data:
    data = coff2_data[key]
    if(not key == "0"):
        avg = sum(data) / len(data)
        coff2_data[key].clear
        coff2_data[key] = avg

for key in coff3_data:
    data = coff3_data[key]
    if(not key == "0"):
        avg = sum(data) / len(data)
        coff3_data[key].clear
        coff3_data[key] = avg
keys =list(coff1_data.keys())
values = list(coff1_data.values())
plt.plot(keys[1:],values[1:], label= "Cof 1")
plt.legend()
plt.title("COFF 1")
plt.show()
keys =list(coff2_data.keys())
values = list(coff2_data.values())
plt.plot(keys[1:],values[1:], label= "Cof 2")
plt.legend()
plt.title("COFF 2")
plt.show()
keys =list(coff3_data.keys())
values = list(coff3_data.values())
plt.plot(keys[1:],values[1:], label= "Cof 3")
plt.legend()
plt.title("COFF 3")
plt.show()
print("end")