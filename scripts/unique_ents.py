import sys,os,json,requests


entityset = set()
count = 0
with open("../data/wikidata_valid_only1.txt") as infile:
    for line in infile:
        count += 1
        if count%1000 == 0:
            print(count)
        s,p,o = line.strip().split("\t")
        entityset.add(s)
        entityset.add(o)

f = open('unique_entities_valid_only1.txt','w')
for entity in entityset:
    f.write("%s\n"%(entity))
f.close()
        
        
