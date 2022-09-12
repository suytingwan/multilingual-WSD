import urllib2
import urllib
import json
import gzip

from StringIO import StringIO

service_url = 'https://babelnet.io/v5/getSynset'

# split in ['de', 'fr', 'it', 'es']

split = 'de'

ids = []
for line in open('./mulan-{}/{}_inventory_train_filter.txt'.format(split, split)):
    bn = line.strip().split('\t')
    ids.extend(bn[1:])

print(len(ids))
keys  = ['aaa']

fw = open('./mulan-{}/{}_bn_gloss.txt'.format(split, split), 'a+')
num_id = 0
num_key = 0

for id in ids:
    if num_id == 1000:
        num_id = 0
        num_key += 1
        print('using key: ', num_key)
    num_id += 1
    key = keys[num_key]

    params = {
        'id' : id,
        'key'  : key,
        'searchLang' : 'EN'
    }

    url = service_url + '?' + urllib.urlencode(params)
    request = urllib2.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib2.urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        buf = StringIO( response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())

        glosses = data['glosses']
        for i, result in enumerate(glosses):
            gloss = result.get('gloss')
            source = result.get('source')
            fw.write(id + '\t' + data['mainSense'].encode('utf-8') + '\t' + source.encode('utf-8') + '\t' + gloss.encode('utf-8') + '\n')
fw.close()

