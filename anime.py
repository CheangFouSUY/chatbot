from jikanpy import Jikan
# import logging

# You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

jikan = Jikan()

MAX_GENRES_SEARCH = 4
INDEX_OF_INFO = 46


def search(patterns, query=""):
    print('Loading...')
    print(patterns)

    # when it is similar
    if len(query):

        isInfo = False
        if int(patterns[0]) >= INDEX_OF_INFO:
            isInfo = True

        patterns = []
        search = jikan.search('anime', query,
                              parameters={'order_by': 'title', 'limit': 1})
        res = search['results']

        # found the id
        for (idx, r) in enumerate(res):
            anime_id = r['mal_id']
            # print(anime_id)

        # check for genres using jika.anime
        anime_info = jikan.anime(anime_id)
        all_genre = anime_info["genres"]
        # print(all_genre)

        # when index is in info intent, go with this one
        print()
        if(isInfo):
            print("Title: " + anime_info['title'])
            print("Description: " + anime_info['synopsis'][0:300] + "...")
            print("Status: " + anime_info['status'])
            print("Genres: ", end="")
            # found all genres in the anime, push the id
            for (idx, r) in enumerate(all_genre):
                print(str(r['name']), end=", ")

        # otherwise, it means just serach similar
        else:
            # found all genres in the anime, push the id
            for (idx, r) in enumerate(all_genre):
                patterns.append(str(r['mal_id']))

            # if too many genres, remove the last few
            if (len(patterns) >= MAX_GENRES_SEARCH):
                patterns = patterns[:len(
                    patterns)-(len(patterns) % MAX_GENRES_SEARCH)]
            print(patterns)
            query = ""
            search = jikan.search('anime', query,
                                  parameters={'genre': ','.join(patterns), 'order_by': 'members', 'limit': 10})
            res = search['results']

            for (idx, r) in enumerate(res):
                print('Title: ' + r['title'])
                print('Description: ' + r['synopsis'][0:100] + '...')
                print()

    # search genre
    else:
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
