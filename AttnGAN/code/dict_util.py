
import json
import matplotlib.pyplot as plt
def save_dict(saveDict,fileName):
    with open(fileName,"a") as csv_file:
        j = json.dumps(saveDict)
        csv_file.write('\r\n')
        csv_file.write(j)
def read_dict(fileName):

    with open(fileName,"r") as csv_file:
        j=csv_file.read()
        readDict = json.loads(s=j)
    return readDict

if __name__ == '__main__':

    mydict = read_dict("medR_salad.csv")
    x = []
    y = []
    for i in mydict:
        x.append(i["ckpt"])
        y.append(i['medR'])
    plt.ylabel('medR')
    plt.xlabel('batch')
    plt.plot(x,y)
    plt.show()
