#importing the requisites for the project
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import schedule
import time
import pandas as pd
import numpy as np

def check_flights():
    url = "https://www.google.com/flights/explore/#explore;f=SFO;t=r-Europe-0x46ed8886cfadda85%253A0x72ef99e6b3fcf079;li=11;lx=13;d=2018-01-16"
    driver = webdriver.PhantomJS()
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36")
    driver = webdriver.PhantomJS(desired_capabilities=dcap,service_args=['--ignore-ssl-errors=true'])
    driver.implicitly_wait(20)
    driver.get(url)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR,"span.CTPFVNB-v-c")))

    bs = BeautifulSoup(driver.page_source, "lxml")

    best_price_tags = bs.findAll('div', 'CTPFVNB-w-e')

    #check if scrap works - alert if it fails and shutdown
    if len(best_price_tags) < 4:
        print("Failed to load Page Data .....")
        requests.post('https://maker.ifttt.com/trigger/fare_alert/with/key/mMEbbDPNhUUaXVZEeG9z-eUTNAZarLxUHg-L1qljjIg',
                    data = {"value1":"script","value2":"failed", "value3":""})
        sys.exit(0)

    else:
        print("Successfully loaded the page data")

    best_prices = []
    for tag in best_price_tags:
        best_prices.append(int(tag.text.replace('$','').replace(',','')))
    best_price = best_prices[0]

    best_heigth_tags = bs.findAll('div', 'CTPFVNB-w-f')
    best_heights = []
    for t in best_heigth_tags:
        best_heights.append(float(t.attrs['style'].split('height:')[1].replace('px;','')))

    best_height = best_heights[0]

    pph = best_price/best_height

    cities = bs.findAll('div', 'CTPFVNB-w-o')
    hlist = []
    for city in cities[0].findAll('div','CTPFVNB-w-x'):
        hlist.append(float(city['style'].split('height:')[1].replace('px;',''))*pph)

    fares = pd.DataFrame(hlist, columns=['price'])

    px = [x for x in fares['price']]
    ff = pd.DataFrame(px, columns=['fare']).reset_index()
    X = StandardScaler().fit_transform(ff)
    db = DBSCAN(eps = 0.5, min_samples=1).fit(X)

    labels = db.labels_
    unique_labels = set(labels)
    clusters = len(unique_labels)

    pf = pd.concat([ff, pd.DataFrame(db.labels_,columns=['cluster'])], axis=1)
    rf = pf.groupby('cluster')['fare'].agg(['min','count'])

    if clusters > 1 \
        and ff['fare'].min() == rf.iloc[0]['min'] \
        and rf.iloc[0]['count'] < rf['count'].quantile(.10) \
        and rf.iloc[0]['fare'] + 100 < rf.iloc[1]['fare']:
            city = bs.find('span', 'CTPFVNB-v-c').text()
            fare = bs.find('span', 'CTPFVNB-v-k').text()
            requests.post('https://maker.ifttt.com/trigger/fare_alert/with/key/mMEbbDPNhUUaXVZEeG9z-eUTNAZarLxUHg-L1qljjIg',
                        data = {"value1":city,"value2":fare, "value3":""})
    else:
        print("No Alert")

    #set schedule to run pour code in every 60 minutes
    schedule.every(60).minutes.do(check_flights)

    while 1:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    print("Successfully Build")
    check_flights()
