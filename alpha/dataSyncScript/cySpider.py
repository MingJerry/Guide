# coding:utf-8
import requests
import argparse
import os
import time
import shutil
import glob
from bs4 import BeautifulSoup


class MakeFolder(object):
    def __init__(self, disable=False):
        self.disable_mkdir = disable
        # self.base_url = "https://www.msdmanuals.com/"
        self.base_url = "https://www.chunyuyisheng.com"
        self.url = "https://www.chunyuyisheng.com/pc/qalist/"
        self.kv = {'user-agent': 'Mozilla/5.0', 'Connection': 'close'}
        self.path = os.getcwd()
        self.root = self.path + self.url.split('/')[-1] + "\\dataSource\\CyQaList\\"
        self.next_page_url = ''

    def make_root_folder(self):
        try:
            if not os.path.exists(self.root):
                os.mkdir(self.root)
            # if not os.path.exists(self.path):
            #     # r = requests.get(url, headers=kv)
            #     r = requests.get(self.url)
            #     with open(self.path, 'wb') as f:
            #         r.raise_for_status()
            #         r.encoding = r.apparent_encoding
            #         f.write(r.content)
            #         print(r.text[1000:2000])
            #         f.close()
            #         print("文件保存成功")
            else:
                shutil.rmtree(self.root, True)
                os.mkdir(self.root)
                print("%s 目录重建" % self.root)

        except Exception as e:
            print("建立目录失败 %s " % self.root)

    def make_child_folder(self):
        save_root = self.root
        # \\dataSource\\CyQaList
        r_request = requests.get(self.url)
        print(r_request.status_code)
        r_text = r_request.text
        sym_soup = BeautifulSoup(r_text, 'html.parser')
        # for sym_name in sym_soup.find_all(attrs='symptoms__main--title'):
        for section in sym_soup.find_all(attrs='tab-item'):
            # sym_soup.find_all(attrs='tab-item')[1].find('a').string
            section_name = section.find('a').string.strip()
            section_path = save_root + section_name + "\\"
            if not os.path.exists(section_path):
                os.mkdir(section_path)
                print("%s 目录建立" % section_name)
            else:
                print("%s 目录已存在" % section_name)
            # for one_sym_name in sym_name
            if section.find('a').attrs['href'].startswith('http'):
                topic_url = section.find('a').attrs['href']
            else:
                topic_url = self.base_url + section.find('a').attrs['href']
            self.get_topic_resolve(next_url=topic_url, next_path=section_path)

    def get_topic_resolve(self, next_url, next_path):
        while not self.next_page_url == next_url:
            if self.next_page_url == next_url:
                break
            save_topic_path = next_path + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.html'
            # 优化为时间格式html
            time.sleep(5)
            r_topic = requests.get(next_url, headers={'Connection': 'close'})
            topic_text = r_topic.text
            topic_soup = BeautifulSoup(topic_text, 'html.parser')

            if not os.path.exists(save_topic_path):
                with open(save_topic_path, 'wb') as topic_file:
                    r_topic.raise_for_status()
                    r_topic.encoding = r_topic.apparent_encoding
                    topic_file.write(r_topic.content)
                    print(r_topic.text[1000:2000])
                    topic_file.close()
                    print("%s 数据存入" % (next_path.split('/')[-1]))
            else:
                print("%s 数据已存在" % (next_path.split('/')[-1]))

            self.next_page_url = next_url
            if topic_soup.find(attrs='next')['href'].startswith('http'):
                next_url = topic_soup.find(attrs='next')['href']
            else:
                next_url = self.base_url + topic_soup.find(attrs='next')['href']


class DataProcess(object):
    def __init__(self):
        self.import_root = os.getcwd() + "\\dataSource\\CyQaList"
        self.import_A_root = os.getcwd() + "\\dataSource\\CyAList"
        self.import_Q_root = os.getcwd() + "\\dataSource\\CyQList"
        self.ele_q_name = "qa-item qa-item-ask"
        self.ele_a_name = "qa-item qa-item-answer"
        print(self.import_root)

    # def resolve_q_data(self):
    #     # -*- coding : utf-8-*-
    #     child_list = os.listdir(self.import_root)
    #     for child in child_list:
    #         child_list_path = os.path.join(self.import_root, child)
    #         html_list = os.listdir(child_list_path)
    #         i = 0
    #         for f_html in html_list:
    #             f_path = os.path.join(child_list_path, f_html)
    #             try:
    #                 f = open(f_path, 'r', encoding='utf-8')
    #                 f_content = f.read()
    #                 f.close()
    #                 data_soup = BeautifulSoup(f_content, 'html.parser')
    #                 # test_quanshen = data_soup.find_all('article')
    #                 for question in data_soup.find_all(attrs='qa-item qa-item-ask'):
    #                     txt_name = "%014d.txt" % i
    #                     txt_path = os.path.join(child_list_path, txt_name)
    #                     f_txt = open(txt_path, "w+", encoding='utf-8')
    #                     f_txt.write(question.text.strip().split('\t')[0])
    #                     print(question.text.strip().split('\t')[0])
    #                     f_txt.write(question.text.strip().split('\t')[-1])
    #                     f_txt.write('\n')
    #                     print(question.text.strip().split('\t')[-1]+'\n')
    #                     f_txt.close()
    #                     i += 1
    #
    #             except IOError as e:
    #                 print(e)

    def resolve_data(self, ele_name="qa-item qa-item-ask"):
        # -*- coding : utf-8-*-
        child_list = os.listdir(self.import_root)
        for child in child_list:
            child_list_path = os.path.join(self.import_root, child)
            old_txt_path = child_list_path + "\\" + "*.txt"
            old_txt_list = glob.glob(old_txt_path)
            for old_txt in old_txt_list:
                print("删除 %s" % old_txt)
                os.remove(old_txt)
            html_list = os.listdir(child_list_path)
            i = 0
            for f_html in html_list:
                f_path = os.path.join(child_list_path, f_html)
                try:
                    f = open(f_path, 'r', encoding='utf-8')
                    f_content = f.read()
                    f.close()
                    data_soup = BeautifulSoup(f_content, 'html.parser')
                    # test_quanshen = data_soup.find_all('article')
                    for question in data_soup.find_all(attrs=ele_name):
                        txt_name = "%08d.txt" % i
                        txt_path = os.path.join(child_list_path, txt_name)
                        f_txt = open(txt_path, "w+", encoding='utf-8')
                        f_txt.write(question.text.strip().split('\t')[0])
                        print(question.text.strip().split('\t')[0])
                        f_txt.write(question.text.strip().split('\t')[-1])
                        f_txt.write('\n')
                        print(question.text.strip().split('\t')[-1]+'\n')
                        f_txt.close()
                        i += 1

                except IOError as e:
                    print(e)

    def resolve_qa_data(self, ele_name="hot-qa-item"):
        # -*- coding : utf-8-*-
        child_list = os.listdir(self.import_root)
        for child in child_list:
            child_list_path = os.path.join(self.import_root, child)
            old_txt_path = child_list_path + "\\" + "*.txt"
            old_txt_list = glob.glob(old_txt_path)
            for old_txt in old_txt_list:
                print("删除 %s" % old_txt)
                os.remove(old_txt)
            html_list = os.listdir(child_list_path)
            i = 0
            for f_html in html_list:
                f_path = os.path.join(child_list_path, f_html)
                try:
                    f = open(f_path, 'r', encoding='utf-8')
                    f_content = f.read()
                    f.close()
                    data_soup = BeautifulSoup(f_content, 'html.parser')
                    # test_quanshen = data_soup.find_all('article')
                    for question in data_soup.find_all(attrs=ele_name):
                        txt_name = "%08d.txt" % i
                        txt_path = os.path.join(child_list_path, txt_name)
                        qa_text_list = question.text.strip().split('\t')
                        while '' in qa_text_list:
                            qa_text_list.remove('')
                        while '\n' in qa_text_list:
                            qa_text_list.remove('\n')
                        q_key = qa_text_list[0].strip() + '\n'
                        q_value = qa_text_list[1].strip() + '\n'
                        a_key = qa_text_list[2].strip() + '\n'
                        a_value = qa_text_list[3].strip() + '\n'
                        # parse = question.text.strip().split('\t').filter(None)
                        f_txt = open(txt_path, "w+", encoding='utf-8')
                        f_txt.write(q_key)
                        print(q_key)
                        f_txt.write(q_value)
                        print(q_value)
                        f_txt.write(a_key)
                        print(a_key)
                        f_txt.write(a_value)
                        print(a_value)
                        f_txt.close()
                        i += 1

                except IOError as e:
                    print(e)


def main(args):
    if args.enable:
        mf = MakeFolder()
        mf.make_root_folder()
        mf.make_child_folder()
    dp = DataProcess()
    # dp.resolve_data('qa-item qa-item-answer')
    dp.resolve_qa_data()
    # dp.resolve_a_data()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", '--enable', help="disable mkdir folder", action='store_true')
    cmd_args = parser.parse_args()
    err_code = main(cmd_args)
    exit(err_code)
