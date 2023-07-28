import sys,os,json,requests

entd = {}
print("reading unique entities")
with open("../data/unique_entities_valid_only1.txt") as infile:
    for line in infile:
        ent = line.strip()
        entd[ent] = ''
print("finished reading unique entities %d"%(len(entd)))
print("fetching descriptions")
with open("../data/wikidata5m_text.txt") as infile:
    for line in infile:
        try:
            entdesc = line.strip().split('\t')
            entd[entdesc[0]] = [" ".join(entdesc[1:])]
        except Exception as err:
            print(err)
print("finished fetching descriptions")

f = open("../data/unique_valid_descriptions1.jsonlines",'w')
for ent,desc in entd.items():
    if not desc:
        continue
    f.write(json.dumps({'entity':ent, 'description':desc})+'\n')
f.close()

        
        
