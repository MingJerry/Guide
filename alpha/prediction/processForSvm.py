import os
import pickle as pk
import jieba as jb
import keras
import re
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from wordcloud import WordCloud
from sqlalchemy import create_engine
from dataSyncScript.config.db_config import mysql_db_config
from logger import alpha_logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


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

        self.clinic_code_T = {value: key for key, value in self.clinic_code.items()}

        self.sql_select_qa = """SELECT * FROM dataManage_qalist WHERE CLINIC != "全部科室" """

        alpha_logger.info("DataPreProcess loading...\n----------------------")
        self.qa = pd.read_sql(sql=self.sql_select_qa, con=self.conn)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('max_colwidth', 100)
        # matplotlib.rcParams['font.family'] = 'serif'

    def show_data_message(self):
        clinic_count = {'Clinic': self.qa['CLINIC'].value_counts().index, 'Count': self.qa['CLINIC'].value_counts()}
        df_clinic = pd.DataFrame(data=clinic_count).reset_index(drop=True)
        alpha_logger.info(df_clinic)
        df_clinic.plot(x='Clinic', y='Count', kind='bar', legend=False, figsize=(8, 5))
        plt.title("科室问题统计")
        plt.ylabel('Count', fontsize=18)
        plt.xlabel('Clinic', fontsize=18)

    def before_process(self):
        alpha_logger.info("训练数据总量：%d" % len(self.qa))
        alpha_logger.info(self.qa.sample(10))
        self.show_data_message()
        return self.divide_and_generate()

    # 定义删除除字母,数字，汉字以外的所有符号的函数
    @staticmethod
    def remove_punctuation(line):
        line = str(line)
        if line.strip() == '':
            return ''
        rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
        line = rule.sub('', line)
        return line

    # 停用词列表
    @staticmethod
    def stop_words_list(file_path):
        stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    @staticmethod
    def generate_wordcloud(tup):
        wordcloud = WordCloud(background_color='white',
                              font_path='simhei.ttf',
                              max_words=50, max_font_size=40,
                              random_state=42
                              ).generate(str(tup))
        return wordcloud

    def word_cloud(self, df):
        cli_text = dict()
        tag_to_clinic = dict(zip(df.Q_Tag, df.CLINIC))
        for cli in self.qa['CLINIC'].value_counts().index:
            text = df.loc[df['CLINIC'] == cli, 'Q_D']
            text = (' '.join(map(str, text))).split(' ')
            cli_text[cli] = text

        fig, axes = plt.subplots(4, 4, figsize=(30, 38))
        k = 1
        for i in range(4):
            for j in range(4):
                cat = tag_to_clinic[k]
                most100 = Counter(cli_text[cat]).most_common(100)
                ax = axes[i, j]
                ax.imshow(self.generate_wordcloud(most100), interpolation="bilinear")
                ax.axis('off')
                ax.set_title("{} Top 100".format(cat), fontsize=10)
                k += 1
                if k == 15:
                    break

            if k == 15:
                break

        plt.show()
        alpha_logger.info("Word Cloud Completed")

    def divide_and_generate(self):
        q_all_list = []
        stop_words = self.stop_words_list("./chineseStopWords.txt")
        qa = self.qa.copy(deep=True)
        qa["Q_Clean"] = qa["QUESTION"].apply(self.remove_punctuation)
        qa["Q_D"] = qa["Q_Clean"].apply(lambda x: " ".join([w for w in jb.lcut_for_search(x) if w not in stop_words]))
        qa["Q_Tag"] = qa["CLINIC"].apply(lambda x: self.clinic_code[x] if x in self.clinic_code
                                         else 0)
        # 生成词云
        self.word_cloud(qa)

        tf_idf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
        features = tf_idf.fit_transform(qa.Q_D)
        labels = qa.Q_Tag
        alpha_logger.info(features.shape)
        alpha_logger.info(features)

        N = 2
        for cli, cli_tag in self.clinic_code.items():
            features_chi2 = chi2(features, labels == cli_tag)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tf_idf.get_feature_names())[indices]
            uni_grams = [v for v in feature_names if len(v.split(' ')) == 1]
            bi_grams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(cli))
            print("  . Most correlated uni-grams:\n       . {}".format('\n       . '.join(uni_grams[-N:])))
            print("  . Most correlated bi-grams:\n       . {}".format('\n       . '.join(bi_grams[-N:])))

        alpha_logger.info("相关性展示")
        return qa, features, labels

    @staticmethod
    def model_train(data):
        alpha_logger.info("Model Train Config loading...")

        X_train, X_test, y_train, y_test = train_test_split(data['Q_D'], data['Q_Tag'], random_state=0,
                                                            stratify=data['Q_Tag'])
        count_vec = CountVectorizer()
        X_train_counts = count_vec.fit_transform(raw_documents=X_train)
        tf_idf_transformer = TfidfTransformer()
        X_train_tf_idf = tf_idf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tf_idf, y_train)

        alpha_logger.info("Model Train Config load Completed.")

        return clf, count_vec

    @staticmethod
    def plot_model(history):
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

    def test_model(self, model, test, source_text, i):
        test_x, test_y = test
        print(source_text.iloc[i]["QUESTION"])
        predict_clinic = model.predict(test_x)
        print("Real Label: ", source_text.iloc[i]["CLINIC"],
              "Predict Label: ", self.clinic_code_T[np.argmax(predict_clinic[i])])


class SVMTrigger(object):
    def __init__(self, model_tmp, vec_tmp):
        self.model_ = model_tmp
        self.vec_ = vec_tmp
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

    # 定义删除除字母,数字，汉字以外的所有符号的函数
    @staticmethod
    def remove_punctuation(line):
        line = str(line)
        if line.strip() == '':
            return ''
        rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
        line = rule.sub('', line)
        return line

    # 停用词列表
    @staticmethod
    def stop_words_list(file_path):
        stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    def pre_for_input(self, input_x):
        print(input_x)

        format_input = " ".join([w for w in jb.lcut_for_search(self.remove_punctuation(input_x))
                                 if w not in self.stop_words_list("./chineseStopWords.txt")])
        pre_cli_tag = self.model_.predict(self.vec_.transform([format_input]))
        print(self.pre_clinic_code_T[pre_cli_tag[0]])

        return 0

    def multi_classify(self, features, labels, df):
        models = [
            # RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
            LinearSVC(),
            # MultinomialNB(),
            LogisticRegression(random_state=0),
        ]
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()

        print(cv_df.groupby('model_name').accuracy.mean())

        # 训练模型
        model = LinearSVC()
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                         test_size=0.33,
                                                                                         stratify=labels,
                                                                                         random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 生成混淆矩阵
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=self.pre_clinic_code.keys(), yticklabels=self.pre_clinic_code.keys())
        plt.ylabel('实际结果', fontsize=18)
        plt.xlabel('预测结果', fontsize=18)
        plt.show()
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=[k for k in self.pre_clinic_code.keys()][:-1]))


if __name__ == '__main__':
    # 分词
    data_pre = DataPreProcess(mysql_db_config)
    train_data, feature_, labels_ = data_pre.before_process()
    model_x, vec = data_pre.model_train(train_data)

    trigger = SVMTrigger(model_x, vec)
    trigger.pre_for_input("手淫导致尿痛是尿道炎吗?")
    trigger.multi_classify(feature_, labels_, train_data)





