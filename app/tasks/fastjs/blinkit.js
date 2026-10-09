(async () => {
  const r = await fetch('/v1/layout/search?q=' + encodeURIComponent(%(q)s) + '&search_type=type_to_search', {
    method: 'POST', body: '{}',
    headers: {'content-type': 'application/json', 'lat': %(lat)s, 'lon': %(lon)s, 'app_client': 'consumer_web'}});
  if (!r.ok) return {status: r.status};
  const j = await r.json();
  const snips = (j.response && j.response.snippets) || [];
  const t = (x) => (x && (x.text || x.title && x.title.text)) || null;
  return {status: r.status, products: snips.filter(s => s.data && s.data.name).map(s => ({
    name: t(s.data.name), pack: t(s.data.variant), price: t(s.data.normal_price), mrp: t(s.data.mrp),
    available: typeof s.data.inventory === 'number' ? s.data.inventory > 0 : true, id: s.data.identity && s.data.identity.id}))};
})()
