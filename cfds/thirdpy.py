import re
import socks
import socket
import requests
import urllib2
import sys, getopt
from bs4 import BeautifulSoup


class DeepWebCrawler:
    pattern = '''href=["'](.[^"']+)["']'''
    domain_regex = r'[http,https]://(.+\.onion)'
    crawled_urls = []
    crawled_domains = []
    counter = 0
    quick_mode = False
    seed = ''
    verbose = False
    output_filename = ''

    def __init__(self, seed, quick_mode=True, verbose=False):

        self.init_socket()

        self.seed = seed
        self.quick_mode = quick_mode
        self.verbose = verbose

        self.output_filename = 'crawled.txt'
        with open(self.output_filename, 'w') as f:
            f.write('Seed ' + seed + '\n')

    def init_socket(self):
        def create_connection(addr, timeout=None, src=None):
            sock = socks.socksocket()
            sock.connect(addr)
            return sock

        # Set our proxy to TOR
        socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, '127.0.0.1', 9050)
        socket.socket = socks.socksocket
        socket.create_connection = create_connection  # force the socket class to use our new create_connection()

    def req(self, url):
        if not (url.endswith('.onion') or url.endswith('/') or url.endswith('.html') or url.endswith(
                '.php') or url.endswith('.asp') or url.endswith('.htm') or url.endswith('.aspx') or url.endswith(
                '.xml') or url.endswith('.jsp') or url.endswith('.jspx') or url.endswith('.txt') or url.endswith(
                'Main_Page')):
            return ''
        proxy_support = urllib2.ProxyHandler({"socks5": "127.0.0.1:9050"})
        opener = urllib2.build_opener(proxy_support)
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        if self.verbose:
            print
            ' [+] Opening URL...'
        return opener.open(url).read()

    def crawl(self, url, action=None, depth=0):

        if self.verbose:
            print
            ' [+] Crawling ' + url
        self.crawled_urls.append(url)
        try:
            request = self.req(url)
        except Exception as e:
            if self.verbose:
                print
                ' [-] ' + str(e)
            return
        if self.verbose:
            print
            ' [+] URL is reachable'

        infotolog = action(url, request)

        if '.onion' in url:
            domain = re.search(self.domain_regex, url).group(1)
            if domain not in self.crawled_domains:
                self.crawled_domains.append(domain)
                self.counter += 1
                print
                ' [+] ' + str(self.counter) + ' domains crawled'
        else:
            domain = ''

        if len(infotolog) > 0:
            log = infotolog + ' @ ' + domain + ' @ Depth ' + str(depth)
            with open(self.output_filename, "a") as f:
                f.write(log + '\n')
            if self.verbose:
                print
                ' [+] Found ' + log

        for nexturl in re.findall(self.pattern, request, re.L):
            if nexturl in self.crawled_urls:
                continue
            new_domain = re.search(self.domain_regex, nexturl)
            if new_domain == None:
                continue
            new_domain = new_domain.group(1)
            if new_domain == domain:
                if url != nexturl and not self.quick_mode:
                    self.crawl(nexturl, action, depth)
            else:
                self.crawl(nexturl, action, depth + 1)


def main():
    verbose = False
    quick_mode = True
    seed = 'http://thehiddenwiki.org/'

    Crawler = DeepWebCrawler(seed, quick_mode, verbose)

    Crawler.crawl(seed, action=Crawler.getTitle)


if __name__ == '__main__':
    main()
