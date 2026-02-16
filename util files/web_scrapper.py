"""
Asynchronous web scraper for Prime Robotics website.

This module scrapes structured question-answer content from:
- Home page
- About page
- Trainee/Courses page

The extracted data is formatted into dictionaries and written
to a JSON file for later use (e.g., model fine-tuning).

Output
------
A JSON file containing a list of:
    {
        "question": str,
        "answer": str
    }

Dependencies
------------
- aiohttp
- asyncio
- beautifulsoup4
"""

import bs4
import aiohttp
import asyncio
import json
from typing import Any, Dict, List


HOME: str = "https://primerobotics.com.ng/"
ABOUT: str = "https://primerobotics.com.ng/about"
COURSES: str = "https://primerobotics.com.ng/trainee"
JSON_FILE: str = "primerobotics.json"
TEXT_FILE: str = "html_content.txt"


def formatted_dict(question: str, answer: str) -> Dict[str, str]:
    """
    Format a question-answer pair into a dictionary.

    Parameters
    ----------
    question : str
        The question text.
    answer : str
        The answer text.

    Returns
    -------
    Dict[str, str]
        Dictionary containing the question and answer.
    """
    return {
        "question": question,
        "answer": answer,
    }


async def fetch_home(session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    """
    Scrape the Prime Robotics home page.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Active HTTP session.

    Returns
    -------
    List[Dict[str, str]]
        Extracted question-answer pairs from the homepage.
    """
    async with session.get(HOME) as response:
        html: str = await response.text()
        data: List[Dict[str, str]] = []
        soup = bs4.BeautifulSoup(html, "html.parser")

        txt = soup.find(class_="text-lg md:text-xl text-blue-100 max-w-2xl")
        if txt:
            data.append(
                formatted_dict("Goal of Prime robotics?", txt.text.strip())
            )

        txt = soup.find("p", class_="text-sm text-blue-200")
        if txt:
            data.append(
                formatted_dict("What prime robotics offer?", txt.text.strip())
            )

        offers = soup.find_all(class_="font-semibold text-lg mb-2")
        offer_text: str = ". ".join(offer.text.strip() for offer in offers)
        if offer_text:
            data.append(
                formatted_dict("4 Steps to get certified?", offer_text)
            )

        contact_list: List[str] = [
            "email",
            "phone number",
            "address",
            "program time",
        ]

        contacts = soup.find_all(class_="flex items-start")
        for index, contact in enumerate(contacts):
            if contact and index < len(contact_list):
                data.append(
                    formatted_dict(
                        f"Contact us: {contact_list[index]}?",
                        contact.text.strip(),
                    )
                )

        return data


async def fetch_trainee(
    session: aiohttp.ClientSession,
) -> List[Dict[str, str]]:
    """
    Scrape the trainee/courses page.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Active HTTP session.

    Returns
    -------
    List[Dict[str, str]]
        Extracted question-answer pairs from courses page.
    """
    async with session.get(COURSES) as response:
        html: str = await response.text()
        data: List[Dict[str, str]] = []
        soup = bs4.BeautifulSoup(html, "html.parser")

        courses = soup.find_all("div", class_="p-6")
        course_list: List[str] = []

        for course in courses:
            title = course.find(
                "h3",
                class_="text-xl font-bold text-blue-800",
            ) or course.find(
                "h3",
                class_="text-xl font-bold text-blue-800 mb-3",
            )

            if not title:
                continue

            title_text: str = title.text.strip()
            course_list.append(title_text)

            details = course.find_all(
                "p", class_="font-semibold text-blue-600"
            )
            if len(details) >= 2:
                duration, price = details[:2]
                data.append(
                    formatted_dict(
                        f"{title_text} Tuition Fee?",
                        price.text.strip(),
                    )
                )
                data.append(
                    formatted_dict(
                        f"{title_text} Duration?",
                        duration.text.strip(),
                    )
                )

            brief_description = course.find(
                "p", class_="text-gray-600 mb-4"
            )
            if brief_description:
                data.append(
                    formatted_dict(
                        f"{title_text} brief description?",
                        brief_description.text.strip(),
                    )
                )

            summary_items = course.find_all(
                "li", class_="flex items-start"
            )
            summary = ". ".join(
                item.text.strip() for item in summary_items
            )
            if summary:
                data.append(
                    formatted_dict(f"{title_text} summary?", summary)
                )

        if course_list:
            data.append(
                formatted_dict(
                    "Available courses?",
                    ". ".join(course_list),
                )
            )

        return data


async def fetch_about(
    session: aiohttp.ClientSession,
) -> List[Dict[str, str]]:
    """
    Scrape the about page.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Active HTTP session.

    Returns
    -------
    List[Dict[str, str]]
        Extracted question-answer pairs from about page.
    """
    async with session.get(ABOUT) as response:
        html: str = await response.text()
        data: List[Dict[str, str]] = []
        soup = bs4.BeautifulSoup(html, "html.parser")

        about = soup.find("p", class_="text-lg text-gray-600")
        if about:
            data.append(
                formatted_dict(
                    "Who is prime robotics?",
                    about.text.strip(),
                )
            )

        return data


async def main() -> None:
    """
    Main asynchronous entry point.

    Workflow
    --------
    1. Create HTTP session.
    2. Run scraping coroutines concurrently.
    3. Aggregate results.
    4. Write results to JSON file.

    Raises
    ------
    aiohttp.ClientError
        If HTTP-related errors occur.
    """
    async with aiohttp.ClientSession() as session:
        try:
            methods = [fetch_home, fetch_about, fetch_trainee]
            tasks: List[List[Dict[str, str]]] = await asyncio.gather(
                *(method(session) for method in methods)
            )

            data: List[Dict[str, str]] = []
            for task in tasks:
                data.extend(task)

        except aiohttp.ClientError as e:
            print(f"HTTP Client Error: {e}")
            return

        else:
            with open(JSON_FILE, "w") as json_file:
                json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
