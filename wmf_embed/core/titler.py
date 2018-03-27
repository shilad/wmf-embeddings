import logging

import pandas as pd


class Titler(object):
    def __init__(self, path):
        logging.info('reading titles from %s', path)
        titles_df = pd.read_csv(path)
        self.titles = {}
        for i, row in titles_df.iterrows():
            if row['project'] == 'concept':
                self.titles['c:' + str(row['page_id'])] = row['title']
        logging.info('reading %d titles', len(self.titles))

    def get_title(self, id):
        if id in self.titles:
            return self.titles[id]
        else:
            return id.split(':', 1)[-1]