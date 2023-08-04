import sys,os,json,glob,random


prop_labels = {}
with open('../wikidata5m_relation.txt') as infile:
    for line in infile:
        line = line.strip()
        items = line.split("\t")
        relationid = items[0]
        relationlabels = items[1:]
        prop_labels[relationid] = relationlabels


master_dict = {}
for file_path in glob.glob("../../scripts/newids/*.json"):
    print("Reading ",file_path)
    d = json.loads(open(file_path).read())
    master_dict.update(d)
    print(len(master_dict))

f = open(sys.argv[1]).readlines()
fw = open(sys.argv[2],'w')
for line in f:
    line = line.strip()
    s,p,o,q = line.split('\t')
    if s not in master_dict or o not in master_dict or p not in prop_labels:
        continue
    print(s,p,o,q)
    print(master_dict[s])
    chain = master_dict[s]['label']+':'+master_dict[s]['newid'] + ' -- ' + p+':'+prop_labels[p][0] + ' -- '+master_dict[o]['label']+':'+master_dict[o]['newid']
    sp = master_dict[s]['label']+':'+master_dict[s]['newid'] + ' -- ' + p+':'+prop_labels[p][0]
    sp = sp.replace(':',' : ')
    answer = master_dict[o]['label']+' : '+master_dict[o]['newid']
    chain = chain.replace(':',' : ')
    jsonline = {'question':q, 'spo':[s,p,o], 'chain':chain, 'answer':answer, 'sp':sp}
    fw.write(json.dumps(jsonline)+'\n')
fw.close()
