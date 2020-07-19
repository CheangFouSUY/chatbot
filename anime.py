from jikanpy import Jikan
# import logging

# You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

jikan = Jikan()


def search(patterns):
    print('Loading...')
    search = jikan.search('anime', '',
                          parameters={'genre': ','.join(patterns), 'order_by': 'members', 'limit': 10})
    res = search['results']

    print()
    for (idx, r) in enumerate(res):
        print('Title: ' + r['title'])
        print('Description: ' + r['synopsis'][0:100] + '...')
        print()

    return 'test'
