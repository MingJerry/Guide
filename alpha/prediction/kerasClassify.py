from keras import models
import pickle as pk
import jieba as jb
import keras
import os
import numpy as np
import datetime
from dataManage.models import QaAdmin, QuestionDemo
from sqlalchemy import create_engine
from dataSyncScript.config.db_config import mysql_db_config
from logger import alpha_logger


class Trigger(object):
    def __init__(self, q_tmp):
        print(os.path.join(os.getcwd(), "prediction\\kara_model_q.h5"))
        self.token = self.load_token()
        self.model = self.load_model()

        self.q = q_tmp
        self.dt = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.p = 1

        self.pre_clinic_code = {"中医科": 1,
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

        self.pre_clinic_code_T = {value: key for key, value in self.pre_clinic_code.items()}

    @staticmethod
    def load_token():
        pkl_file = open(os.path.join(os.getcwd(), "prediction\\token_pk.pkl"), 'rb')
        token_file = pk.load(pkl_file)
        pkl_file.close()
        return token_file

    @staticmethod
    def load_model():
        question_model = models.load_model(os.path.join(os.getcwd(), "prediction\\kara_model_q.h5"))
        return question_model

    def pre_for_input(self):
        print(self.q)
        input_list = [jb.lcut_for_search(self.q)]
        input_x_seq = self.token.texts_to_sequences(input_list)
        print(input_x_seq)
        input_x_pad = keras.preprocessing.sequence.pad_sequences(
            input_x_seq, padding='post', truncating='post', maxlen=50
        )
        print(input_x_pad)
        output_x = self.model.predict(input_x_pad)
        output_y = self.pre_clinic_code_T[np.argmax(output_x)]
        print("Predict Output:", output_y)
        self.save_in_question_demo(self.q, output_y, self.dt, self.p)
        return output_y

    @staticmethod
    def save_in_question_demo(q_, a_, d_, p_):
        QuestionDemo.objects.get_or_create(QUESTION_USER=q_, ANSWER_USER=a_, QUEST_DATETIME=d_, PRE_METHOD=p_)


def save_in_qa_admin(q_, a_):
    d_ = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    QaAdmin.objects.get_or_create(QUESTION_ADMIN=q_, ANSWER_ADMIN=a_, QA_DATETIME=d_)


if __name__ == '__main__':
    trigger = Trigger('手淫导致尿痛是尿道炎吗?')
    trigger.pre_for_input()
