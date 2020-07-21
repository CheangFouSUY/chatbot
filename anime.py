from jikanpy import Jikan
# import logging

# You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

jikan = Jikan()


def search(patterns, query=""):
    print('Loading...')
    # when it is similar
    if len(query):
        patterns = []
        search = jikan.search('anime', query,
                              parameters={'order_by': 'title', 'limit': 1})
        res = search['results']

        # found the id
        for (idx, r) in enumerate(res):
            anime_id = r['mal_id']
            # print(anime_id)

        # check for genres using jika.anime
        all_genre = jikan.anime(anime_id)["genres"]

        # found all genres in the anime, push the id
        for (idx, r) in enumerate(all_genre):
            patterns.append(str(r['mal_id']))
            # print(patterns)
        query = ""

    # search genre
    search = jikan.search('anime', query,
                          parameters={'genre': ','.join(patterns), 'order_by': 'members', 'limit': 10})
    res = search['results']

    print()
    for (idx, r) in enumerate(res):
        print('Title: ' + r['title'])
        print('Description: ' + r['synopsis'][0:100] + '...')
        print()

    if (len(res) == 0):
        print("No results found")

    return 'test'
