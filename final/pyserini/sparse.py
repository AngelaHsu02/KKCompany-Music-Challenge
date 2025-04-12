import pandas as pd
from tqdm import tqdm

# index, session_id, song_id, unix_played_at, play_status, login_type, listening_order
train_source = pd.read_parquet("../../datagame-2023/label_train_source.parquet")
# index, session_id, song_id, unix_played_at, play_status, login_type, listening_order
train_target = pd.read_parquet("../../datagame-2023/label_train_target.parquet")
# index, session_id, song_id, unix_played_at, play_status, login_type, listening_order
test_source = pd.read_parquet("../../datagame-2023/label_test_source.parquet")
# index, song_id, artist_id, song_length, album_id, language_id, album_month
meta_song = pd.read_parquet("../../datagame-2023/meta_song.parquet")
# index, song_id, composer_id
meta_song_composer = pd.read_parquet("../../datagame-2023/meta_song_composer.parquet")
# index, song_id, genre_id
meta_song_genre = pd.read_parquet("../../datagame-2023/meta_song_genre.parquet")
# index, song_id, lyricist_id
meta_song_lyricist = pd.read_parquet("../../datagame-2023/meta_song_lyricist.parquet")
# index, song_id, producer_id
meta_song_producer = pd.read_parquet("../../datagame-2023/meta_song_producer.parquet")
# index, song_id, title_text_id
meta_song_titletext = pd.read_parquet("../../datagame-2023/meta_song_titletext.parquet")

# Preparse session's songs
from collections import defaultdict

session_to_songs = defaultdict(list)  # key -> session_id, value -> songs

test_source = test_source.sort_values(by=['session_id', 'listening_order'])

group_by_session = test_source.groupby('session_id')

for _, group_song in tqdm(group_by_session):
    session_id = group_song['session_id'].iloc[0]
    session_to_songs[session_id] = group_song['song_id'].tolist()

session_to_time = dict()
for row in tqdm(train_target.itertuples(), total=len(train_target)):
    session_to_time[row.session_id] = row.unix_played_at
for row in tqdm(test_source.itertuples(), total=len(test_source)):
    session_to_time[row.session_id] = row.unix_played_at

# Preparse session's songs
from collections import defaultdict

session_to_songs = defaultdict(list)  # key -> session_id, value -> songs

test_source = test_source.sort_values(by=['session_id', 'listening_order'])

group_by_session = test_source.groupby('session_id')

for _, group_song in tqdm(group_by_session):
    session_id = group_song['session_id'].iloc[0]
    session_to_songs[session_id] = group_song['song_id'].tolist()

""" For Jelinek-Mercer smoothing """
from pyserini.search.lucene import LuceneSearcher


class MyLuceneSearcher(LuceneSearcher):
    def set_jmlm(self, Lambda: float = 0.9999):
        """ Set the Jelinek-Mercer smoothing with lambda

        Reference java code:
            public void set_qld(float mu) {
              this.similarity = new LMDirichletSimilarity(mu); # SimpleSearcher

              // We need to re-initialize the searcher
              searcher = new IndexSearcher(reader); # SimpleSearcher.searcher
              searcher.setSimilarity(similarity); # SimpleSearcher.searcher
            }
        Args:
            l (float): Lamda
        """
        from jnius import autoclass

        LMDirichletSimilarity = autoclass("org.apache.lucene.search.similarities.LMJelinekMercerSimilarity")
        self.object.similarity = LMDirichletSimilarity(Lambda)

        # We need to re-initialize the searcher
        IndexSearcher = autoclass("org.apache.lucene.search.IndexSearcher")
        self.object.searcher = IndexSearcher(self.object.reader)
        self.object.searcher.setSimilarity(self.object.similarity)
        print("set to jmlm with lambda = {}".format(Lambda))

from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher, querybuilder
from pyserini.analysis import get_lucene_analyzer

fields = ['artist', 'album', 'language', 'genre']


class Searcher():
    def __init__(self, searcher: LuceneSearcher, reader: IndexReader, is_stemming=False) -> None:
        self.searcher: LuceneSearcher = searcher
        self.searcher.set_analyzer(get_lucene_analyzer(stemming=is_stemming))
        self.total_docs = reader.stats()['documents']
        self.mu = reader.stats()['total_terms'] / reader.stats()['unique_terms']

    def song_to_contents(self, song_id):
        return self.searcher.doc(song_id).contents() if self.searcher.doc(song_id) else ""

    def songs_to_query(self, session, time_range, song_ids, cut_song_token=0):
        time_min = session_to_time[session] - time_range
        time_max = session_to_time[session] + time_range
        contents = [self.song_to_contents(song_id) for song_id in song_ids]
        contents = [content.split() for content in contents]

        query_text = []
        query_text_len = 0
        count = 10

        for content in contents:
            max_song_token = count
            is_first = True
            first_index = -1
            last_index = -1
            for i in range(len(content)):
                if max_song_token <= 0:
                    break
                if any(field in content[i] for field in fields):
                    query_text.append(content[i])
                    query_text_len += len(content[i])
                    continue
                int_x = session_to_time[int(content[i], 16)]
                if time_min < int_x < time_max:
                    if is_first:
                        is_first = False
                        first_index = i
                    last_index = i
                    query_text.append(content[i])
                    query_text_len += len(content[i])
                    max_song_token -= 1

                if int_x > time_max:
                    if is_first:
                        is_first = False
                        first_index = i - 1
                        last_index = i - 1
                    break

            for i in range(first_index, -1, -1):
                if max_song_token <= 0 or any(field in content[i] for field in fields):
                    break
                query_text.append(content[i])
                query_text_len += len(content[i])
                max_song_token -= 1

            for i in range(last_index + 1, len(content)):
                if max_song_token <= 0 or any(field in content[i] for field in fields):
                    break
                query_text.append(content[i])
                query_text_len += len(content[i])
                max_song_token -= 1

        query_text = " ".join(query_text)

        return query_text

    def set_max_clause_count(self, max_clause_count):
        print(type(self.searcher.object.searcher))
        print("Original maxCaluseCount:", self.searcher.object.searcher.maxClauseCount)
        self.searcher.object.searcher.setMaxClauseCount(max_clause_count)
        print("Updated maxCaluseCount:", self.searcher.object.searcher.maxClauseCount)

    def search(self, queries, args):
        # self.searcher.set_bm25(b=0, k1=0)
        # self.searcher.set_qld(self.mu)
        self.searcher.set_jmlm()

        results = []

        for session_id, qtext in tqdm(queries):
            hits = self.searcher.search(qtext, args.k)
            results.append([session_id, [hit.docid for hit in hits]])
        return results

class Arg:
    def __init__(self, k):
        self.k = k


args = Arg(k=100)
index = "indexes/collection_jsonl_sparse"
stem = False

luceneSearcher = MyLuceneSearcher(index)
reader = IndexReader(index)
searcher = Searcher(searcher=luceneSearcher, reader=reader, is_stemming=stem)

max_clause_count = 1000000
searcher.set_max_clause_count(max_clause_count)

# Prepare queries = [[session_id, query], ...]
print("Preparing queries...")
last_n_song = 10  # TODO : check this
queries = []
too_long_count = 0
for session, songs in tqdm(session_to_songs.items()):
    qtext = searcher.songs_to_query(session, 0, songs[-5:])
    queries.append([session, qtext])
print("Searching...")
results = searcher.search(queries, args)

import pickle

with open('jmlm_0.9999_token10.pkl', 'wb') as file:
    pickle.dump(results, file)