import pandas 
import requests
from bs4 import BeautifulSoup

r = requests.get("https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds")
html = r.text
soup = BeautifulSoup(html, "html.parser")


firstH3 = soup.find('h3') # Start here
uls = []
for nextSibling in firstH3.findNextSiblings():
    if nextSibling.name == 'h2':
        break
    if nextSibling.name == 'ul':
        uls.append(nextSibling)

etfs = []
for ul in uls:
    for li in ul.findAll('li'):
        
        a = li.text
        a = li.text.split('(')
        a = li.text.split(')')
        a = a[0].split()
        etfs.append(a[-1])
        
#print(len(etfs))
"""
soup = BeautifulSoup(html, "html.parser")
data1 = soup.find('ul')
for li in data1.find_all("li"):
    soup2 = BeautifulSoup(li.text, "html.parser")
    data2 = soup2.find('li')
    print(type(li.text))
    for li2 in data2.find_all("a"):
        print(li2.text, end=" ")"""