import requests
import argparse
import os
from bs4 import BeautifulSoup
from time import sleep


class MakeFolder(object):
    def __init__(self, disable=False):
        self.disable_mkdir = disable
        self.base_url = "https://www.msdmanuals.com/"
        self.url = "https://www.msdmanuals.com/zh/首页/symptoms"
        self.kv = {'user-agent': 'Mozilla/5.0', 'Connection': 'close'}
        self.root = "C://N-5CG8020SV9-Data//xsu//Documents//XM//Guide//alpha//dataSyncScript//dataSource//Symptoms//"
        self.path = self.root + self.url.split('/')[-1] + ".html"

    def get_html(self):
        try:
            if not os.path.exists(self.root):
                os.mkdir(self.root)
            if not os.path.exists(self.path):
                # r = requests.get(url, headers=kv)
                r = requests.get(self.url)
                with open(self.path, 'wb') as f:
                    r.raise_for_status()
                    r.encoding = r.apparent_encoding
                    f.write(r.content)
                    print(r.text[1000:2000])
                    f.close()
                    print("文件保存成功")
            else:
                print("文件已存在")

        except Exception as e:
            print("爬取失败")

    def resolve_html(self):
        save_path = self.root
        # "C://N-5CG8020SV9-Data//xsu//Documents//XM//Guide//alpha//dataSyncScript//dataSource//Symptoms//"
        # sleep(5)
        r_request = requests.get(self.url)
        print(r_request.status_code)
        r_text = r_request.text
        sym_soup = BeautifulSoup(r_text, 'html.parser')
        # for sym_name in sym_soup.find_all(attrs='symptoms__main--title'):
        for section in sym_soup.find_all(
                attrs='symptoms__container symptoms__section-list-view symptoms__container--open'
        ):
            section_name = section.find('h2').string
            section_path = save_path + section_name + '//'
            if not os.path.exists(section_path):
                os.mkdir(section_path)
                print("%s 目录建立" % section_name)
            else:
                print("%s 目录已存在" % section_name)
            # for one_sym_name in sym_name

            for item in section.find_all('a'):
                item_name = item.string
                item_path = section_path + item_name + '//'
                if not os.path.exists(item_path):
                    os.mkdir(item_path)
                    print("%s 目录建立" % item_name)
                else:
                    print("%s 目录已存在" % item_name)
                if item.attrs['href'].startswith('http'):
                    topic_url = item.attrs['href']
                else:
                    topic_url = self.base_url + item.attrs['href']
                self._get_topic_resolve(next_url=topic_url, next_path=item_path)

    @staticmethod
    def _get_topic_resolve(next_url, next_path):
        save_topic_path = next_path + next_path.split('/')[-3] + '.html'
        sleep(5)
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


class DataProcess(object):
    def __init__(self):
        self.path = os.getcwd() + "\\dataSource\\Symptoms" + "\\全身\\呃逆\\呃逆.html"
        print(self.path)

    def resolve_data(self):
        try:
            f = open(self.path, 'r', encoding='utf-8')
            f_content = f.read()
            f.close()
            data_soup = BeautifulSoup(f_content, 'html.parser')
            # test_quanshen = data_soup.find_all('article')
            for topic_header in data_soup.find_all(attrs='topic__header--section'):
                print(topic_header.text.strip())
        except IOError as e:
            print(e)


def main(args):
    if args is not None:
        # mf = MakeFolder()
        # mf.get_html()
        # mf.resolve_html()
        dp = DataProcess()
        dp.resolve_data()
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", '--disable', help="disable mkdir folder", action='store_true')
    cmd_args = parser.parse_args()
    err_code = main(cmd_args)
    exit(err_code)
