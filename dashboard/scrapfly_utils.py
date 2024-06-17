# This file modified from https://github.com/scrapfly/scrapfly-scrapers/tree/main/tiktok-scraper
# In compliance with that license, the contents of this file are available under the NPOSL-3.0 license terms as well.

from scrapfly import ScrapeConfig, ScrapflyClient, ScrapeApiResponse
import jmespath
from typing import Dict, List
import json

with open("scrapfly_key.txt", "r") as key_file:
    SCRAPFLY_API_KEY = key_file.read().strip()

SCRAPFLY = ScrapflyClient(key=SCRAPFLY_API_KEY)

BASE_CONFIG = {
    # bypass tiktok.com web scraping blocking
    "asp": True,
    # set the proxy country to US
    "country": "US",
}


def parse_post(response: ScrapeApiResponse) -> Dict:
    """parse hidden post data from HTML"""
    selector = response.selector
    data = selector.xpath("//script[@id='__UNIVERSAL_DATA_FOR_REHYDRATION__']/text()").get()
    post_data = json.loads(data)["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
    parsed_post_data = jmespath.search(
        """{
        id: id,
        desc: desc,
        createTime: createTime,
        video: video.{duration:duration, ratio: ratio, cover: cover, playAddr: playAddr, downloadAddr: downloadAddr, bitrate: bitrate},
        author: author.{id: id, uniqueId: uniqueId, nickname: nickname, avatarLarger: avatarLarger, signature: signature, verified: verified},
        stats: stats,
        locationCreated: locationCreated,
        diversificationLabels: diversificationLabels,
        suggestedWords: suggestedWords,
        contents: contents[].{textExtra: textExtra[].{hashtagName: hashtagName}}
        }""",
        post_data,
    )
    return parsed_post_data


async def scrape_posts(urls: List[str]) -> List[Dict]:
    """scrape tiktok posts data from their URLs"""
    to_scrape = [ScrapeConfig(url, **BASE_CONFIG) for url in urls]
    data = []
    async for response in SCRAPFLY.concurrent_scrape(to_scrape):
        post_data = parse_post(response)
        data.append(post_data)
    return data[0]["video"]["cover"]
