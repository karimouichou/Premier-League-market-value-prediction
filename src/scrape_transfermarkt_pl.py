"""
Scrape Premier League 2025/26 squads + market values from Transfermarkt.

Run LOCALLY (not in a sandbox):
    pip install requests beautifulsoup4 pandas lxml
    python scrape_transfermarkt_pl.py

Output: pl_2025_26_market_values.csv

NOTE: Be polite. Transfermarkt's terms restrict automated scraping, so keep the
delay, scrape once, cache the result, and use it for personal/learning purposes.
"""
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.transfermarkt.com"
SEASON = 2025  # Transfermarkt uses the START year: 2025 = 2025/26
SEASON_END = pd.Timestamp("2026-05-31")  # reference date for computing age
DELAY = 4      # seconds between requests
HEADERS = {
    # Without a browser-like User-Agent, Transfermarkt returns 403
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)


def get_soup(url):
    time.sleep(DELAY)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_value(text):
    """'€45.00m' -> 45_000_000 ; '€900k' -> 900_000 ; '-' -> None"""
    text = text.strip().replace("€", "")
    m = re.match(r"([\d.,]+)\s*(m|k|bn)?", text)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    mult = {"m": 1e6, "k": 1e3, "bn": 1e9}.get(m.group(2), 1)
    return int(num * mult)


def get_clubs():
    url = f"{BASE}/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={SEASON}"
    soup = get_soup(url)
    clubs = {}
    for a in soup.select("table.items td.hauptlink a[href*='/startseite/verein/']"):
        m = re.search(r"/([^/]+)/startseite/verein/(\d+)", a["href"])
        if m:
            clubs[a.get_text(strip=True)] = (m.group(1), m.group(2))
    return clubs


def get_squad(club_name, slug, club_id):
    url = f"{BASE}/{slug}/kader/verein/{club_id}/saison_id/{SEASON}/plus/1"
    soup = get_soup(url)
    rows = []
    for tr in soup.select("table.items > tbody > tr"):
        name_a = tr.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not name_a:
            continue
        player_id = re.search(r"/spieler/(\d+)", name_a["href"]).group(1)
        cells = tr.find_all("td", recursive=False)
        position = tr.select_one("table.inline-table tr:nth-of-type(2) td")
        value_cell = tr.select_one("td.rechts.hauptlink")

        # Find the date-of-birth cell by content, not by column index.
        # Format on the page: "Aug 11, 1999 (26)"
        dob, age_on_page = None, None
        for c in cells:
            m = re.match(r"([A-Z][a-z]{2} \d{1,2}, \d{4})\s*\((\d+)\)", c.get_text(strip=True))
            if m:
                dob = pd.to_datetime(m.group(1), format="%b %d, %Y")
                age_on_page = int(m.group(2))
                break

        rows.append({
            "player_id": player_id,
            "player": name_a.get_text(strip=True),
            "club": club_name,
            "position": position.get_text(strip=True) if position else None,
            "dob": dob,
            "age_on_page": age_on_page,
            "age_season_end": (
                round((SEASON_END - dob).days / 365.25, 1) if dob is not None else None
            ),
            "market_value_eur": parse_value(value_cell.get_text()) if value_cell else None,
        })
    return rows


def main():
    clubs = get_clubs()
    print(f"Found {len(clubs)} clubs")
    all_rows = []
    for name, (slug, cid) in clubs.items():
        print(f"Scraping {name}...")
        all_rows.extend(get_squad(name, slug, cid))
    df = pd.DataFrame(all_rows)
    df.to_csv("pl_2025_26_market_values.csv", index=False)
    print(f"Saved {len(df)} players")


if __name__ == "__main__":
    main()
