import sys,os,json,glob,random


prop_labels = {}
with open('../data/wikidata5m_relation.txt') as infile:
    for line in infile:
        line = line.strip()
        items = line.split("\t")
        relationid = items[0]
        relationlabels = items[1:]
        prop_labels[relationid] = relationlabels


master_dict = {}
for file_path in glob.glob("newids/*.json"):
    if 'sample.json' in file_path:
        continue
    print("Reading ",file_path)
    d = json.loads(open(file_path).read())
    master_dict.update(d)
    print(len(master_dict))

f = open('../data/pre-train-data.txt','w')
with open('../data/wikidata_valid_only1.txt') as infile:
    count = 0
    for line in infile:
        count += 1
        try:
            s,p,o = line.strip().split("\t")
            triple = master_dict[s]['label']+' : '+' : '.join(master_dict[s]['newid'].split(':'))+' --  '+p+' : '+random.choice(prop_labels[p])+' -- '+master_dict[o]['label']+' : '+' : '.join(master_dict[o]['newid'].split(':'))
            if count%10000 == 0:
                print(triple)
                print(count)
            f.write(triple+'\n')
        except Exception as err:
            pass
f.close()

