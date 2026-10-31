"""Every icon the page asks for exists, is a real image, and is served.

The brand mark and the favicons are referenced by hand-written URLs in
``index.html`` and in the manifest, and nothing else in the suite would notice a
renamed file, a wrong MIME type, or a file the sandbox filesystem truncated on
write (which it did to two of these on their first copy).  So: parse the links
out of the markup, check the bytes decode, and check the extensions the server
claims to know.
"""
import json
import os
import re
import sys

_WEB = os.path.dirname(__import__("pyroorb.web", fromlist=["_"]).__file__)
_STATIC = os.path.join(_WEB, "static")
_ICONS = os.path.join(_STATIC, "icons")


def _referenced(name):
    html = open(os.path.join(_STATIC, name), encoding="utf-8").read()
    return re.findall(r'href="(/static/icons/[^"]+)"', html)


def _local(url):
    assert url.startswith("/static/")
    return os.path.join(_STATIC, url[len("/static/"):])


def test_every_referenced_icon_exists_and_decodes():
    from PIL import Image
    urls = _referenced("index.html") + _referenced("memory.html")
    assert urls, "the page stopped referencing its icons"
    for url in urls:
        path = _local(url)
        assert os.path.isfile(path), url
        if path.endswith(".webmanifest"):
            continue
        img = Image.open(path)
        img.load()                       # a truncated write raises here
        assert img.size[0] == img.size[1]


def test_the_ico_carries_the_three_legacy_sizes():
    from PIL import IcoImagePlugin
    ico = IcoImagePlugin.IcoFile(open(os.path.join(_ICONS, "favicon.ico"), "rb"))
    assert {(16, 16), (32, 32), (48, 48)} <= set(ico.sizes())


def test_the_manifest_points_at_files_that_exist():
    doc = json.load(open(os.path.join(_ICONS, "site.webmanifest"), encoding="utf-8"))
    assert doc["icons"], "a manifest with no icons installs nothing"
    for entry in doc["icons"]:
        assert os.path.isfile(_local(entry["src"])), entry["src"]


def test_the_server_knows_the_image_types():
    from pyroorb.web import server
    for ext in (".png", ".ico", ".webmanifest"):
        assert ext in server._CTYPE
    assert server._CTYPE[".png"] == "image/png"
