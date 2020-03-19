# coding:utf-8
import os
import pandas as pd
import datetime
import argparse
from dataSyncScript.config.db_config import mysql_db_config
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


class DataImport(object):
    def __init__(self, db_config, db_driver='mysql'):
        self.engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s?charset=utf8") % (db_config.USER_NAME,
                                                                                                  db_config.PASSWORD,
                                                                                                  db_config.HOST,
                                                                                                  db_config.DBNAME))
        self.conn = self.engine.connect()
        self.cur_path = os.getcwd()
        self.data_root = os.path.join(self.cur_path, "dataSource\\CyQaList")
        print(self.data_root)

    def source_data(self):
        fold_list = os.listdir(self.data_root)
        first_create = True
        i = 0
        for fold in fold_list:
            fold_path = os.path.join(self.data_root, fold)
            file_list = os.listdir(fold_path)
            for file in file_list:
                file_path = os.path.join(fold_path, file)
                df_txt = pd.read_csv(file_path, sep='\n', header=None)
                print(file_path)
                df_txt = df_txt.drop(index=[0, 2]).T
                df_txt.columns = ['QUESTION', 'ANSWER']
                col_name_tmp = df_txt.columns.tolist()
                col_name_tmp.insert(0, 'QAINDEX')
                col_name_tmp.insert(1, 'CLINIC')
                col_name_tmp.insert(4, 'SNAPSHOT')
                # print(col_name_tmp)
                df_txt = df_txt.reindex(columns=col_name_tmp)
                df_txt['QAINDEX'] = i
                i += 1
                df_txt['CLINIC'] = fold
                df_txt['SNAPSHOT'] = datetime.date.today()
                # print(df_txt)
                df_txt.to_sql(name="clinic_data_source", con=self.conn,
                              schema='ALPHA_TEST', index=False,
                              if_exists='replace' if first_create else "append")
                first_create = False
        # release_fpb.to_sql(name="brace_preprocessed_sub_feature", con=self.conn,
        #                    schema='common_dashboard',
        #                    if_exists='replace' if self.truncate else "append")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", '--disable', help="disable mkdir folder", action='store_true')
    cmd_args = parser.parse_args()
    DI = DataImport(mysql_db_config)
    DI.source_data()

