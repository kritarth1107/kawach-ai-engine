(async () => {
  const r = await fetch('/v1/layout/search?q=' + encodeURIComponent(%(q)s) + '&search_type=type_to_search', {
    method: 'POST', body: '{}',
    headers: {'content-type': 'application/json', 'lat': %(lat)s, 'lon': %(lon)s, 'app_client': 'consumer_web'}});
  if (!r.ok) return {status: r.status};
  const j = await r.json();
  const snips = (j.response && j.response.snippets) || [];
  const t = (x) => (x && (x.text || x.title && x.title.text)) || null;
  // logged in on this profile already? (the web app keeps its access token in localStorage 'auth') — no code needed then
  let li = false; try { li = !!(JSON.parse(localStorage.getItem('auth') || '{}') || {}).accessToken; } catch (e) {}
  return {status: r.status, logged_in: li, products: snips.filter(s => s.data && s.data.name).map(s => ({
    name: t(s.data.name), pack: t(s.data.variant), price: t(s.data.normal_price), mrp: t(s.data.mrp),
    available: typeof s.data.inventory === 'number' ? s.data.inventory > 0 : true, id: s.data.identity && s.data.identity.id,
    // the item as the web app keeps it in its own cart (localStorage 'cart'): the saved cart step writes it there too
    cart_item: (() => { const c = ((s.data.atc_action || {}).add_to_cart || {}).cart_item; return c ? {product_id: c.product_id, price: c.price, mrp: c.mrp,
      unit: c.unit, group_id: c.group_id, image_url: c.image_url, merchant_id: c.merchant_id} : null; })()}))};
})()
