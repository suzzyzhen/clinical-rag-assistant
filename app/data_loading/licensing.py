"""Extract licence evidence, not a determination of permission to redistribute."""

import re
from urllib.parse import urljoin

CC_URL = re.compile(
    r"https?://creativecommons\.org/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(\d\.\d)(/igo)?/?",
    re.I,
)
CC_NAME = re.compile(r"\bCC\s+BY(?:-NC)?(?:-ND|-SA)?\s+\d\.\d(?:\s+IGO)?\b", re.I)


def unknown_license():
    return dict(
        license=None,
        license_evidence=None,
        license_source=None,
        license_detection="unknown",
    )


def detect_license(evidence, source, detection="explicit"):
    result = unknown_license()
    if not evidence:
        return result
    text = re.sub(r"\s+", " ", evidence).strip()
    urls = list(CC_URL.finditer(text))
    names = {m.group().upper() for m in CC_NAME.finditer(text)}
    names.update(f"CC {m[1].upper()} {m[2]}" + (" IGO" if m[3] else "") for m in urls)
    result.update(
        license_evidence=text, license_source=source, license_detection="unrecognized"
    )
    if len(names) == 1:
        result.update(
            license=names.pop(),
            license_detection=detection,
        )
    elif len(names) > 1:
        result["license_detection"] = "ambiguous"
    return result


def iris_license(metadata, source):
    values = [
        entry.get("value", "")
        for key, entries in metadata.items()
        if key.lower().startswith(("dc.rights", "dcterms.license", "dc.license"))
        for entry in entries
    ]
    return detect_license("\n".join(values), source)


def html_license(soup, source):
    evidence = []
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").lower()
        if name in {
            "dc.rights",
            "dc.rights.license",
            "dc.rights.uri",
            "dcterms.license.uri",
            "cc:license",
            "dcterms.rights",
            "dcterms.license",
            "license",
            "copyright",
        }:
            evidence.append(tag.get("content", ""))
    for tag in soup.find_all(["a", "link"], href=True):
        if "license" in [rel.lower() for rel in tag.get("rel", [])]:
            evidence.append(
                tag.get_text(" ", strip=True) + " " + urljoin(source, tag["href"])
            )
    result = detect_license("\n".join(evidence), source)
    if result["license_detection"] == "unknown":
        text = soup.get_text(" ", strip=True)
        match = re.search(r"licensed under|work is available under", text, re.I)
        if match:
            result = detect_license(
                text[max(0, match.start() - 100) : match.end() + 350],
                source,
                "candidate",
            )
    # Keep one policy reference only when no licence notice was found.
    if result["license_detection"] != "unknown":
        return result
    for tag in soup.find_all("a", href=True):
        href = urljoin(source, tag["href"])
        label = tag.get_text(" ", strip=True)
        if not href.startswith(("https://", "http://")):
            continue
        policy = href + " " + label
        specific = re.search(
            r"copyright|permissions?(?:[ /_-]+and)?[ /_-]+licen[sc]ing", policy, re.I
        )
        if specific or re.search(r"terms[ /_-]+of[ /_-]+use", policy, re.I):
            result.update(
                license_source=href,
                license_evidence=label or href,
                license_detection="policy_link",
            )
            if specific:
                break

    return result


def pdf_license(pdf, source):
    notices = []
    for number, page in enumerate(pdf, 1):
        text = page.get_text()
        # Keep nearby context so exceptions and referenced licences can be reviewed.
        for match in re.finditer(
            r"creativecommons\.org/licenses/|\bCC\s+BY|licensed under|work is available under",
            text,
            re.I,
        ):
            notices.append(
                f"Page {number}: "
                + text[max(0, match.start() - 200) : match.end() + 350]
            )
    return detect_license("\n".join(notices), source, "candidate")
