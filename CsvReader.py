import csv

def getCsvAsList(filaname):
    with open(filaname + '.csv', 'r') as csvout:
        csvreader = csv.reader(csvout, delimiter=',')
        csvlist = list(csvreader)
        csvlist.pop(0) #removes the explaination line
    return csvlist