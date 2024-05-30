import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':

    url = "https://www.cnblogs.com/huggingface/p/17790578.html"

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # print("html:", response.text)

        # 找到id为"navList"的ul标签
        navs = soup.find("ul", id="navList")
        print("导航栏:")
        li_tags = navs.find_all("li")
        for li_tag in li_tags:
            print(li_tag.get_text().strip(), end=' ')

        title = soup.find('a', attrs={'id': 'Header1_HeaderTitle', 'class': 'headermaintitle HeaderMainTitle'})
        content = soup.find('div', class_="post")
        print(f"标题: {title.text}\n"
              f"内容: {content.text}\n")

    else:
        print("无法获取页面内容")
