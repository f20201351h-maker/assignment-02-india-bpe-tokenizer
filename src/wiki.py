"""Small, polite client for the official MediaWiki Action API.

Only two endpoints are used:
  * action=query&prop=extracts (TextExtracts, plain text) -> page text + revision id
  * action=parse&oldid=REVID                              -> rendered HTML of that exact revision
"""
import json
import time
import urllib.parse

import requests

UA = "india-bpe-tokenizer/1.0 (educational tokenizer project; python-requests)"
_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})


MIN_INTERVAL = 1.0  # seconds between requests (Wikimedia asks clients to be gentle)
_last = [0.0]


def api(lang, params, retries=8):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    p = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = _session.get(url, params=p, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)) + 1)
                continue
            if r.status_code == 200:
                d = r.json()
                if "error" in d and d["error"].get("code") == "maxlag":
                    time.sleep(5)
                    continue
                return d
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API request failed: {lang} {params}")


def fetch_extract(lang, title):
    """Plain-text extract of the *current* revision, plus its revision id."""
    d = api(lang, {
        "action": "query", "prop": "extracts|revisions|info", "titles": title,
        "redirects": "1", "explaintext": "1", "exsectionformat": "plain",
        "rvprop": "ids|timestamp", "inprop": "url",
    })
    page = d["query"]["pages"][0]
    if page.get("missing"):
        return None
    rev = page["revisions"][0]
    return {
        "lang": lang, "title": page["title"], "pageid": page["pageid"],
        "revid": rev["revid"], "rev_timestamp": rev["timestamp"],
        "url": page.get("fullurl"), "text": page.get("extract", ""),
    }


def fetch_html(lang, revid):
    d = api(lang, {"action": "parse", "oldid": str(revid), "prop": "text", "disablelimitreport": "1"})
    return d["parse"]["text"]


def fetch_links(lang, title, limit=5000):
    """Main-namespace article titles linked from a page."""
    out, cont = [], {}
    while len(out) < limit:
        d = api(lang, {"action": "query", "prop": "links", "titles": title, "redirects": "1",
                       "plnamespace": "0", "pllimit": "max", **cont})
        for p in d["query"]["pages"]:
            out += [l["title"] for l in p.get("links", [])]
        if "continue" not in d:
            break
        cont = d["continue"]
    return out[:limit]


def fetch_random_titles(lang, n, seed_note=""):
    out = []
    while len(out) < n:
        d = api(lang, {"action": "query", "list": "random", "rnnamespace": "0", "rnlimit": "max"})
        out += [x["title"] for x in d["query"]["random"]]
    return out[:n]


def page_url(lang, title):
    return f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
