#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os

years = [2025] 
tables = ["advanced", "per_game", "totals"]

os.makedirs("wnba_data/raw_data", exist_ok=True)

for year in years:
    for table in tables:
        url = f"https://www.basketball-reference.com/wnba/years/{year}_{table}.html"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        html_table = soup.find("table", {"id": table})

        rows = []
        for tr in html_table.find("tbody").find_all("tr", class_="full_table"):
            row = {}
            for td in tr.find_all(["th", "td"]):
                stat = td.get("data-stat")
                if stat == "player":
                    # Extract the player name from the <a> tag
                    a_tag = td.find("a")
                    row[stat] = a_tag.get_text(strip=True) if a_tag else td.get_text(strip=True)
                else:
                    row[stat] = td.get_text(strip=True)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"wnba_data/raw_data/{year}_{table}.csv", index=False)
        print(f"{year} {table} saved ({len(df)} rows)")

        time.sleep(4)


# In[2]:


print(df.head())
print(df.shape)
print(df.columns.tolist())


# In[3]:


url = "https://www.spotrac.com/wnba/rankings/salary/"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")
soup


# In[4]:


url = "https://www.basketball-reference.com/wnba/years/2025.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Print all table ids
for table in soup.find_all("table"):
    print(table.get("id"))


# In[5]:


years = [2025]
team_tables = ["wnba_standings", "advanced-team"]
os.makedirs("wnba_data/raw_data", exist_ok=True)

for year in years:
    url = f"https://www.basketball-reference.com/wnba/years/{year}.html"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    for table in team_tables:
        html_table = soup.find("table", {"id": table})

        rows = []
        for tr in html_table.find("tbody").find_all("tr"):
            row = {}
            for td in tr.find_all(["th", "td"]):
                stat = td.get("data-stat")
                row[stat] = td.get_text(strip=True)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"wnba_data/raw_data/{year}_{table}.csv", index=False)
        print(f"{year} {table} saved ({len(df)} rows)")

    time.sleep(4)


# In[6]:


url = "https://herhoopstats.com/salary-cap-sheet/wnba/players/salary_2025/stats_2024/"
page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print(page.status_code)
soup = BeautifulSoup(page.text, "html.parser")
for table in soup.find_all("table"):
    print(table.get("id"), table.get("class"))


# In[8]:


# Print the first row's raw HTML to see the structure
table = soup.find("table", {"class": "salary-stat"})
first_row = table.find("tbody").find("tr")
print(first_row.prettify())


# In[7]:


# Print header row to see column names
header_row = table.find("thead").find("tr")
print(header_row.prettify())


# In[8]:


# Salary data from "Her Hoop Stats WNBA Salary Cap Database" (Add citation)
# All other data is from "Basketball Reference WNBA Season Pages" (Add citation)
salary_years = [2025]

for salary_year in salary_years:
    url = f"https://herhoopstats.com/salary-cap-sheet/wnba/players/salary_{salary_year}/stats_{salary_year}/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    table = soup.find("table", {"class": "salary-stat"})
    if table is None:
        print(f"{salary_year} not found")
        continue

    # get column names from header
    headers = [th.get_text(strip=True) for th in table.find("thead").find("tr").find_all("th")]

    # parse rows
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = tr.find_all("td")
            row = {}
            for i, td in enumerate(cells):
                if i >= len(headers):
                    continue
                col = headers[i]
                if "salary_player_name" in td.get("class", []):
                    a_tag = td.find("a", {"class": "d-none d-sm-block"})
                    row[col] = a_tag.get_text(strip=True) if a_tag else td.get_text(strip=True)
                elif "salary_cap_hit" in td.get("class", []):
                    # extract just the salary from sorttable_customkey "0200000-RFA-PV"
                    key = td.get("sorttable_customkey", "")
                    salary = key.split("-")[0]  # gets "0200000"
                    row[col] = int(salary) if salary.isdigit() else td.get_text(strip=True)
                elif i == 2:
                    # signing status "RFA", "Core", "--"
                    key = td.get("sorttable_customkey", "")
                    row[col] = key.split("-")[0] if key else td.get_text(strip=True)
                else:
                    row[col] = td.get_text(strip=True)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"wnba_data/raw_data/salary_{salary_year}.csv", index=False)
    print(f"salary_{salary_year} saved ({len(df)} rows)")

    time.sleep(3)


# In[9]:


df.head()


# In[ ]:




