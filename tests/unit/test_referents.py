import time

from jarvis.agent import referents


def test_ttl_one_per_kind_no_clear_unrelated():
    store = referents.ReferentStore(ttl_s=600.0)
    cid = "local"
    store.set_result_set(cid, referents.ResultSetRef(id="r1", page_url="http://x", items=[referents.ResultItem(position=1,title="A",host="x",url="http://x/1")], created_at=time.time()))
    store.set_page(cid, referents.PageRef(url="http://x", title="x", origin="http://x", created_at=time.time()))
    r = store.get(cid)
    assert r.result_set is not None
    assert r.page is not None
    # unrelated action should not clear page
    store.set_action(cid, referents.ActionRef(tool="open_app", ok=True, created_at=time.time()))
    r2 = store.get(cid)
    assert r2.page is not None


def test_clear_on_navigation():
    store = referents.ReferentStore(ttl_s=600.0)
    cid = "local"
    store.set_page(cid, referents.PageRef(url="http://a.com", title="a", origin="http://a.com", created_at=time.time()))
    # simulate navigation to different origin
    store.clear_page(cid)
    r = store.get(cid)
    assert r.page is None
