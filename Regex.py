import re

extraction=open("actualdata.txt","r")
listnum= []

for line in extraction:
    line= line.rstrip()
    extract= re.findall("([0-9]+)",line)

    if len(extract) <1: continue

    for i in range(len(extract)):
        num= float(extract[i])
        listnum.append(num)
print(sum(listnum))