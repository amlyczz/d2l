import requests

url = "https://weibo.com/ajax/statuses/mymblog?uid=7707156952&page=0&feature=0"

headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "client-version": "v2.45.97",
    "cookie": "PC_TOKEN=2d40636a22; XSRF-TOKEN=_trnTcTwVtb6hEIAzIdFiP0V; SCF=AiOs4JmA9Zf7cb-b9bynAl7IpXqkrnIksa3zmHU5ufz_U-B9DFD1oMsqgQYNKNKaZ5yR-VRQub02aXjt8GU5gRU.; SUB=_2A25LucArDeRhGeFG6VIQ8yzLwzWIHXVot13jrDV8PUNbmtANLVX5kW9NfggyslVS1Ka1WPnHn8gfW6VjGRPbITrF; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5jJhvHrMjoA04QsRBxR_af5NHD95QN1hz7eKeES0n4Ws4Dqcjci--fiKLhi-2Ri--fiKLhi-iWi--NiKLWiKnXi--Xi-zRiKy2i--fiKLhi-2Ri--fiKLhi-iW; ALF=02_1726299515; WBPSESS=TBR_oyTVulE86-lRnHwg2rtw1naHxI9k8tR9ZoLtJ3Z3-NS47rcTmRmYPe5kf7yl8YzigAn1CTj6EHE0ePaVxqw_7HKfTKTga_idzgON31sFUitIbHPwzWAXyPgPeu9fvjL7fb5wuo9f3wtbF52pvw==; WBPSESS=TBR_oyTVulE86-lRnHwg2q2jDYpu9_UaaS0LXn-zrJpgu8aPR-Bbr8-X00wapxcjpUj7sqV-bxAobT4Ws8KmK6KF5ZLXNdUMGUYXNuRililn7vcEPfm59fxtjEaidvba",
    "priority": "u=1, i",
    "referer": "https://weibo.com/u/7707156952",
    "sec-ch-ua": '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "server-version": "v2024.08.14.1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
    "x-requested-with": "XMLHttpRequest",
    "x-xsrf-token": "_trnTcTwVtb6hEIAzIdFiP0V"
}

response = requests.get(url, headers=headers)

# Print the response in a readable format
print(response.json())

if __name__ == '__main__':
    print()