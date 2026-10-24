from jarvis.browser.sites import DEFAULT_SITES, SiteDirectory, render_site_catalogue, sites_from_settings
from jarvis.config import Settings, SiteSettings


def test_default_sites_present():
    d = sites_from_settings(Settings())
    names = [s.name for s in d.sites]
    for n in ["Google", "YouTube", "Amazon", "Gmail", "Google Maps", "Wikipedia", "GitHub", "Reddit", "Stack Overflow", "LinkedIn", "Spotify web", "Netflix", "X", "Flipkart"]:
        assert n in names


def test_amazon_default_is_amazon_com():
    d = sites_from_settings(Settings())
    s = d.resolve("Amazon") or d.resolve("amazon")
    assert s is not None
    assert s.url == "https://www.amazon.com"


def test_config_overrides_default_by_name():
    d = sites_from_settings(Settings(sites=[SiteSettings(name="YouTube", url="https://yt.example.com")]))
    s = d.resolve("YouTube")
    assert s is not None
    assert s.url == "https://yt.example.com"


def test_search_templates_exist_where_expected():
    d = sites_from_settings(Settings())
    yt = d.resolve("YouTube")
    assert yt is not None and yt.search_url_template and "{q}" in yt.search_url_template
    g = d.resolve("Google")
    assert g is not None and g.search_url_template


def test_catalogue_under_1500():
    d = sites_from_settings(Settings())
    assert len(render_site_catalogue(d)) < 1500
