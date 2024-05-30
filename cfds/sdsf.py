import requests
from bs4 import BeautifulSoup

url = 'http://news.buaa.edu.cn/'

if __name__ == '__main__':

    response = requests.get(url)
    response.encoding = response.apparent_encoding
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    target_div = soup.find('div', {'class': 'main1newsconbot auto'})
    ul_list = target_div.find('ul')

    keyword = "校长"

    matching_li_list = [li for li in ul_list.find_all('li') if keyword in li.text]

    for li in matching_li_list:
        print("校长标签内容：", li.text)
        a_tag = li.find('a')
        if a_tag:
            href = a_tag.get('href')
            print("对应url:", url + href)
