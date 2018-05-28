from urllib.request import urlopen
from multiprocessing.dummy import Pool as ThreadPool

urls = [
  'http://www.python.org',
  'http://www.python.org/about/',
  'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
  'http://www.python.org/doc/',
  'http://www.python.org/download/',
  'http://www.python.org/getit/',
  'http://www.python.org/community/',
  'https://wiki.python.org/moin/',
  'http://planet.python.org/',
  'https://wiki.python.org/moin/LocalUserGroups',
  'http://www.python.org/psf/',
  'http://docs.python.org/devguide/',
  'http://www.python.org/community/awards/'
  # etc..
  ]
another_list = range(len(urls))

def open_url(params):
  url, value = params
  print(url, value)
  urlopen(url)
  return 1

# Make the Pool of workers
pool = ThreadPool(4)
# Open the urls in their own threads
# and return the results
params = zip(urls, another_list)
results = pool.map(open_url, params)
#close the pool and wait for the work to finish
print('Results:', results)
pool.close()
pool.join()