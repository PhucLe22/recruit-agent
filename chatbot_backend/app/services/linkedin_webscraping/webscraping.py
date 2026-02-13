from urllib.parse import urlencode, urljoin
from playwright.sync_api import sync_playwright
from dataclasses import dataclass
import pandas as pd
import logging
import re
from selectolax.parser import HTMLParser
import time
from dotenv import load_dotenv
import os
import random
import pprint
from tqdm import tqdm


username = "DarynBang"

logging.basicConfig(
    level=logging.INFO,  # can be DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()

PAGE_NUMBER = 1

def login_to_linkedin(page, email, password, headless):
    # Go to the LinkedIn login page
    page.goto("https://www.linkedin.com/login")
    page.wait_for_load_state('load')

    page.get_by_label("Email or phone").fill(email)

    logger.info("Filled in Email Successfully!")

    page.get_by_label("Password").fill(password)

    logger.info("Filled in Password Successfully!")

    time.sleep(3)
    page.locator("#organic-div form").get_by_role("button", name="Sign in").click()
    page.wait_for_load_state('load')

    logger.info("Signed in to Linkedin Account Successfully!")
    time.sleep(5)


def extract_job_data(job_url, job_id, job_location, username, unique_id, page):
    """
    Access each job page and extract metadata (title, company, description, etc.).
    You will need to fill in the selectors for HTMLParser.
    """
    job_data = {}
    try:
        # Navigate to job detail page
        full_url = f"{job_url}?currentJobId={job_id}"
        logger.info(f'Currently at Job URL: {full_url}')
        page.goto(full_url, timeout=20_000)
        page.wait_for_load_state("load", timeout=20_000)
        time.sleep(2.5)  # small delay for dynamic content

        html = page.content()
        tree = HTMLParser(html)

        # Fill these out yourself with actual selectors
        job_title = tree.css_first("div.t-24.job-details-jobs-unified-top-card__job-title").text(strip=True)\
            if tree.css_first("div.t-24.job-details-jobs-unified-top-card__job-title") else None

        company_name = tree.css_first("div.job-details-jobs-unified-top-card__company-name").text(strip=True)\
            if tree.css_first("div.job-details-jobs-unified-top-card__company-name") else None

        job_description_container = tree.css_first("article.jobs-description__container")

        if job_description_container:
            description = job_description_container.css_first("div.mt4").text(strip=True)\
                if tree.css_first("div.mt4") else None

            if description:
                logger.info(f"Successfully extracted job description for job: {job_id} - {job_title}! ")

        job_data = {
            "user": username,
            "unique_id": unique_id,
            "job_id": job_id,
            "job_url": full_url,
            "job_location": job_location,
            "job_title": job_title,
            "company_name": company_name,
            "description": description
        }
    except Exception as e:
        logger.error(f"Failed to extract job data for {job_id}: {e}")

    return job_data

def scrape_jobs(page, params, username: str):
    global PAGE_NUMBER
    main_url = "https://www.linkedin.com/jobs/"
    base_url = "https://www.linkedin.com/jobs/search/"
    url = f"{base_url}?{urlencode(params)}"

    job_url_list = []  # store job_url, job_id, location
    job_list = []      # final extracted jobs
    seen_ids = set()   # track unique jobs across searches

    try:
        # Go to the search results page
        page.goto(url, timeout=20_000)
        page.wait_for_load_state("load", timeout=20_000)
    except Exception as e:
        logger.error(f"Failed to load LinkedIn jobs search page: {e}")
        return {}

    while True:
        try:
            # Scroll to load more results
            page.locator("div.scaffold-layout__list").click()
            for _ in range(30):
                page.mouse.wheel(0, 250)
            page.wait_for_timeout(3000)

            time.sleep(2.5)

            # Parse page HTML with Selectolax
            html = page.content()
            tree = HTMLParser(html)

            jobs = tree.css("li.ember-view.uDNkyhNwymJrcaxJBBeOKGNaqEPdqeTOOSQ.occludable-update.p0.relative.scaffold-layout__list-item")
            if jobs:
                logger.info(f"Found {len(jobs)} jobs on current page")

            for job in jobs:
                try:
                    job_link = job.css_first("a")
                    job_id = job.attributes.get("data-occludable-job-id")
                    job_location_node = job.css_first(".job-card-container__metadata-wrapper")
                    job_location = job_location_node.text(strip=True) if job_location_node else None

                    job_url = urljoin(main_url, job_link.attributes.get("href")) if job_link else None

                    if not job_url:
                        logger.warning(f"Job URL was not found - {job_url}, moving to default case! ")
                        job_url = f'https://www.linkedin.com/jobs/view/{job_id}/?alternateChannel=search'
                        logger.info(f'New Job URL: {job_url}')

                    # Unique ID check (username + job_id)
                    unique_id = f"{username}_{job_id}"
                    if unique_id in seen_ids:
                        logger.info(f"Skipping {job_id} as already seen")
                        continue

                    job_url_list.append({
                        "user": username,
                        "unique_id": unique_id,
                        "job_id": job_id,
                        "job_url": job_url,
                        "job_location": job_location,
                    })
                    seen_ids.add(unique_id)

                    logger.info(f"Collected job with unique id: {unique_id} at URL: {job_url}")

                    # Stop once we hit 20 jobs
                    if len(job_url_list) >= 20:
                        break
                except Exception as e:
                    logger.error(f"Error parsing job card: {e}")

            # ✅ Break out of the outer while loop if enough jobs already
            if len(job_url_list) >= 20:
                logger.info("Collected enough jobs, stopping search.")
                break

            # Move to next page if needed
            try:
                PAGE_NUMBER += 1
                next_button = page.get_by_role("button", name=f"Page {PAGE_NUMBER}")
                next_button.click()
                page.wait_for_load_state("load")
            except Exception:
                logger.info("No more pages found.")
                break

        except Exception as e:
            logger.error(f"Error in job scraping loop: {e}")
            break

    # Traverse collected job URLs
    for job_item in tqdm(job_url_list, desc="Extracting Metadata from Job", total=len(job_url_list)):
        data = extract_job_data(job_item["job_url"], job_item["job_id"], job_item["job_location"], job_item['user'], job_item['unique_id'], page)
        if data:
            job_list.append(data)
        time.sleep(random.uniform(0.5, 3.5))

    return {"jobs": job_list, "count": len(job_list)}


def retrieve_linkedin_jobs(headless, params_list):
    LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
    LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
    
    # Return mock data if credentials are missing
    if not LINKEDIN_EMAIL or not LINKEDIN_PASSWORD:
        logger.warning("LinkedIn credentials not found. Returning mock job data.")
        return {
            "jobs": [
                {
                    "job_id": f"mock_job_{i}",
                    "title": f"Software Engineer {i}",
                    "companyName": f"Tech Company {i}",
                    "location": "Ho Chi Minh City",
                    "posted_date": "2025-12-20",
                    "job_link": "https://linkedin.com/jobs/view/mock",
                    "slug": f"software-engineer-{i}-mock-{datetime.now().strftime('%Y%m%d')}"
                }
                for i in range(1, 6)
            ],
            "count": 5,
            "message": "LinkedIn credentials not configured. Showing mock data."
        }

    all_jobs = {"jobs": [], "count": 0}

    # Start browser only once
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--window-size=1366,768",
                "--start-maximized"
            ]
        )

        for params in params_list:
            # Create fresh context for each param set
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/117.0.0.0 Safari/537.36",
                locale="en-US",
                viewport={"width": 1366, "height": 768},
                java_script_enabled=True,
                bypass_csp=True
            )

            page = context.new_page()
            page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1"
            })

            # Login fresh for each context
            login_to_linkedin(page, email=LINKEDIN_EMAIL, password=LINKEDIN_PASSWORD, headless=headless)

            logger.info(f"Crawl starting... Params: {params}")
            result = scrape_jobs(page, params, username=username)

            # Aggregate results
            all_jobs["jobs"].extend(result["jobs"])
            all_jobs["count"] += result["count"]

            # Clean up context
            context.close()

            time.sleep(10)

        browser.close()

    return all_jobs


if __name__ == '__main__':
    params = [
        {"keywords": "AI Engineer", "location": "Ho Chi Minh City"},
        {"keywords": "Backend Developer", "location": "Ho Chi Minh City"},
    ]
    pprint.pprint(retrieve_linkedin_jobs(headless=False, params_list=params))

