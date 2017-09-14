import csv
data=[]
with open('/Users/chiragyeole/Downloads/Sheet_1.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		data.append(row[2])

for i in range(1,80):
	#print(data[i])
	words=data[i].split()
	for j in range(1,len(words)):
		print(words[j])
