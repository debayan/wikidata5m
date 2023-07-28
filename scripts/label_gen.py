import sys,os,json,requests

cache = {}
def sparqlendpoint(entity):
    try:
        url = 'http://localhost:1234/api/endpoint/sparql'
        query = '''PREFIX wd: <http://www.wikidata.org/entity/> select * where { %s ?p ?o } limit 1'''%('wd:'+entity) 
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''

final = []
count = 0
with open("data/wikidata5m_all_triplet.txt") as infile:
    for line in infile:
        count += 1
        if count%1000 == 0:
            print(count)
        try:
            s,p,o = line.strip().split(" ")
        except:
            s,p,o = line.strip().split("\t")
        if s in cache and o in cache:
            final.append([s,p,o])
            print("in cache")
            continue
        result1 = sparqlendpoint(s)
        result2 = sparqlendpoint(o)
        if len(result1['results']['bindings']) != 0 and len(result2['results']['bindings']) != 0:
            final.append([s,p,o])
            cache[s] = 1
            cache[o] = 1
f = open('wikidata_valid_only1.txt','w')
for triple in final:
    f.write("%s\t%s\t%s\n"%(triple[0],triple[1],triple[2]))
f.close()
print("final len",len(final))
        
        
