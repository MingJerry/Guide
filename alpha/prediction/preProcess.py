import os
import urllib
import numpy, keras
import re
from random import randint
import pandas as pd
import jieba as jb
import datetime
import argparse
from dataSyncScript.config.db_config import mysql_db_config
from sqlalchemy import create_engine
from logger import alpha_logger


class DataPreProcess(object):
    def __init__(self, db_config, db_driver='mysql'):
        self.engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s?charset=utf8") % (db_config.USER_NAME,
                                                                                                  db_config.PASSWORD,
                                                                                                  db_config.HOST,
                                                                                                  db_config.DBNAME))
        self.conn = self.engine.connect()
        self.cur_path = os.getcwd()
        alpha_logger.info(self.cur_path)
        self.clinic_one_hot_code = {"中医科": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "产科": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "儿科": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "内科": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "口腔颌面科": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "外科": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "妇科": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    "男科": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    "皮肤性病科": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    "眼科": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    "耳鼻咽喉科": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    "肿瘤及防治科": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    "营养科": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    "骨伤科": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    "全部科室": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
        self.clinic_code = {"中医科": 1,
                            "产科": 2,
                            "儿科": 3,
                            "内科": 4,
                            "口腔颌面科": 5,
                            "外科": 6,
                            "妇科": 7,
                            "男科": 8,
                            "皮肤性病科": 9,
                            "眼科": 10,
                            "耳鼻咽喉科": 11,
                            "肿瘤及防治科": 12,
                            "营养科": 13,
                            "骨伤科": 14,
                            "全部科室": 15}

        self.sql_select_qa = """SELECT * FROM dataManage_qalist"""
        self.qa = pd.read_sql(sql=self.sql_select_qa, con=self.conn)
        alpha_logger.info("DataPreProcess loading...\n----------------------")

    def divide_q(self):
        q_all_list = []
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('max_colwidth', 100)

        qa = self.qa.copy(deep=True)
        qa["Q_D"] = qa["QUESTION"].apply(lambda x: jb.lcut_for_search(x))
        qa["Q_Tag"] = qa["CLINIC"].apply(lambda x: self.clinic_code[x] if x in self.clinic_code
                                         else 0)
        token = keras.preprocessing.text.Tokenizer(num_words=6000)
        q_list = qa["QUESTION"].to_list()
        q_d_list = qa["Q_D"].to_list()
        token.fit_on_texts(q_d_list)

        alpha_logger.info("词索引 ")
        alpha_logger.info(token.word_index)
        alpha_logger.info("词数 ")
        alpha_logger.info(token.word_counts)

        qa["Q_S"] = token.texts_to_sequences(q_d_list)
        q_seq_list = qa["Q_S"].to_list()
        q_seq_array = keras.preprocessing.sequence.pad_sequences(q_seq_list, padding='post', truncating='post', maxlen=50)
        qa["Q_S_padding"] = q_seq_array.tolist()

        clinic_tag_list = qa["Q_Tag"].tolist()
        clinic_tag_array = keras.preprocessing.sequence.utils.to_categorical(clinic_tag_list, num_classes=16)
        qa["Clinic_S"] = clinic_tag_array.tolist()

        alpha_logger.info("\n %s", qa.head(n=2))
        alpha_logger.info("Divide & Tag & Sequence Complete.")
        train_data = (q_seq_array, clinic_tag_array)
        return qa, train_data
        # print(qa.head()

    def model_train(self, n_lay, key_num, word_len, out_lay, train):
        alpha_logger.info("Model Train Config loading...")
        train_x, train_y = train
        model_question = keras.models.Sequential()
        model_question.add(keras.layers.Embedding(output_dim=n_lay,
                                                  input_dim=key_num,
                                                  input_length=word_len))
        model_question.add(keras.layers.Flatten())
        model_question.add(keras.layers.Dense(units=256, activation='relu'))
        model_question.add(keras.layers.Dropout(0.3))
        model_question.add(keras.layers.Dense(units=out_lay, activation='softmax'))
        alpha_logger.info(model_question.summary())
        alpha_logger.info("Model Train Config load Completed.")

        alpha_logger.info("Model Train Starting...")
        model_question.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history_q = model_question.fit(x=train_x, y=train_y, validation_split=0.2, epochs=10, batch_size=128, verbose=1)
        alpha_logger.info("Model Train Completed.")
        self.plot_model(history_q)

    def plot_model(self, history):
        import matplotlib.pyplot as plt

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        acc_values = history.history['acc']
        val_acc_values = history.history['val_acc']

        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        alpha_logger.info("Plot Completed")


if __name__ == '__main__':
    # 分词
    data_pre = DataPreProcess(mysql_db_config)
    qa_, train_data_ = data_pre.divide_q()
    data_pre.model_train(n_lay=32, key_num=6000, word_len=50, out_lay=16, train=train_data_)
    alpha_logger.info("PreProcessing completed.")

