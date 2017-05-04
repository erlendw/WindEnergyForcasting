import csv

def getCsvAsList(filaname):
    with open(filaname + '.csv', 'r') as csvout:
        csvreader = csv.reader(csvout, delimiter=',')
        csvlist = list(csvreader)
        csvlist.pop(0) #removes the explaination line
    return csvlist

def writeToFile(filename, dates, data):
	with open(filename, 'w', newline='') as csvfile:
	    w = csv.writer(csvfile, delimiter=' ',
	                            quotechar=',', quoting=csv.QUOTE_NONE)

	    for i in range(len(data)):
	    	print(dates[i])
	    	w.writerow([dates[i],data[i]])
	    
	    #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
