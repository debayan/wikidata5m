import sys,os,json

d = json.loads(open(sys.argv[1]).read())

count = 0
sampled = {}
for k,v in d.items():
    count += 1
    sampled[k] = v
    if count == 1000:
        break

f = open('newids/sample.json','w')
f.write(json.dumps(sampled,indent=4))
f.close()
