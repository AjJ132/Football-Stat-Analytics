import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
import time
from threading import Lock
import os
import json
from datetime import datetime

class ScraperHelper:
    def __init__(self, save_path, num_workers=None):
        self.save_path = save_path
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.retry_count = 0
        self.max_retries = 3
        self.retry_lock = Lock()
        self.wait_time = 60

    def make_request(self, url, headers=None):
        session = requests.Session()
        while self.retry_count < self.max_retries:
            try:
                if not headers:
                    headers = {
                        'User-Agent': 'Mozilla/5.0',
                        'Cache-Control': 'no-cache'
                    }
                response = session.get(url, headers=headers)
                if response.status_code == 403:
                    with self.retry_lock:
                        if self.retry_count < self.max_retries:
                            print(f"Rate limited. Waiting {self.wait_time/60} minutes. Retry {self.retry_count + 1}/{self.max_retries}")
                            self.retry_count += 1
                            time.sleep(self.wait_time)
                            session.cookies.clear()
                            continue
                        else:
                            raise Exception("Max retries reached. Shutting down.")
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code != 403:
                    raise
        raise Exception("Max retries reached. Shutting down.")

    def get_soup(self, url):
        response = self.make_request(url)
        return BeautifulSoup(response.text, 'html.parser')


    @staticmethod
    def format_date(date_string):
        return parse(date_string).strftime('%Y-%m-%d') if date_string else None