import requests

url = "https://www.msdmanuals.com/zh/首页/symptoms"
kv = {'user-agent': 'Mozilla/5.0'}

try:
    r = requests.get(url, headers=kv)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    print(r.text[1000:2000])

except Exception as e:
    print("爬取失败")
