(async () => {
  // Swiggy Instamart product search, guest, no clicks. Run inside a tab already on https://www.swiggy.com/instamart
  // (after the AWS WAF challenge reload has finished, i.e. window.___INITIAL_STATE___ exists).
  const q = String(%(q)s), lat = Number(%(lat)s), lng = Number(%(lon)s);
  const out = { status: 0, products: [] };
  // The first response for a new browser is an AWS WAF JS challenge page that reloads itself; the real Swiggy pages set
  // window.___INITIAL_STATE___ (Instamart) or window.___INITIAL_STATE__ (food). Give a page that is still loading a moment.
  for (let i = 0; i < 80 && !(window.___INITIAL_STATE___ || window.___INITIAL_STATE__); i++) await new Promise(r => setTimeout(r, 100));
  // "matcher" header the web app sends on every API call (rot-7 of rand5 + Date.now() + rand5).
  const A = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const rot = (t, e) => t.replace(/[a-zA-Z0-9]/g, c => A[(A.indexOf(c) + e + A.length) % A.length]);
  const r5 = () => { const f = (x, n = 0) => n > 10 ? 99999 : (x < 1e4 && (x *= 10), x > 1e4 ? x : f(x, n + 1)); return f(Math.round(Math.random() * 1e5)); };
  const matcher = () => rot(String(r5()) + String(Date.now()) + String(r5()), 7);
  const waf = window.AwsWafIntegration;
  const F = waf && typeof waf.fetch === 'function' ? waf.fetch.bind(waf) : fetch;
  const post = (url, body) => F(url, { method: 'POST', credentials: 'same-origin', body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json', 'matcher': matcher() } });
  const units = m => (m && m.units != null) ? Number(m.units) + (Number(m.nanos || 0) / 1e9) : null;
  try {
    // 1) location -> serving store (pod)
    const r1 = await post('/api/instamart/home/select-location/v2', { data: { lat, lng, clientId: 'INSTAMART-APP' } });
    out.status = r1.status;
    if (!r1.ok) { out.error = 'select-location HTTP ' + r1.status + (r1.headers.get('x-amzn-waf-action') ? ' (waf ' + r1.headers.get('x-amzn-waf-action') + ')' : ''); return out; }
    const j1 = await r1.json();
    const ci = (((j1.data || {}).configs || {}).IM_PAGE_CONFIGS || {}).configInfo;
    const cfg = ci && ci[0] && ci[0].card;
    const pods = (cfg && cfg.podDetailsList) || [];
    const prim = pods.find(p => p.priority === 'PRIORITY_PRIMARY') || pods[0] || (cfg && cfg.podDetails);
    const sec = pods.find(p => p.priority === 'PRIORITY_SECONDARY');
    if (!cfg || !prim || !prim.podId || cfg.serviceabilityStatus !== 'SERVICEABLE') {
      if (j1.statusCode === 0) out.serviceable = false;   // a normal answer with no store: Instamart does not deliver here
      out.error = 'Instamart not serviceable at this location' + (cfg ? ' (' + cfg.serviceabilityStatus + ')' : '');
      return out;
    }
    out.serviceable = true;
    const sla = prim.serviceabilityDetails && prim.serviceabilityDetails.sla;
    if (sla && sla.value) out.eta = sla.value + ' ' + String(sla.unit || 'MINS').toLowerCase();
    out.store = prim.podId;
    // 2) search in that store
    const qs = new URLSearchParams({ offset: '0', ageConsent: 'false', layoutId: String(cfg.layoutId || 4987), voiceSearchTrackingId: '',
      storeId: prim.podId, primaryStoreId: prim.podId, secondaryStoreId: sec ? sec.podId : '' });
    const r2 = await post('/api/instamart/search/v2?' + qs, { facets: [], sortAttribute: '', query: q, search_results_offset: '0',
      page_type: 'INSTAMART_AUTO_SUGGEST_PAGE', is_pre_search_tag: false });
    out.status = r2.status;
    if (!r2.ok) { out.error = 'search HTTP ' + r2.status; return out; }
    const j2 = await r2.json();
    const seen = new Set();
    const take = (it) => {
      for (const v of it.variations || []) {
        const id = v.spinId || v.skuId; if (!id || seen.has(id)) continue; seen.add(id);
        const p = v.price || {};
        const inv = v.inventory || {};
        out.products.push({
          name: v.displayName || it.displayName,
          pack: v.quantityDescription || '',
          price: units(p.offerPrice) > 0 ? units(p.offerPrice) : units(p.mrp),
          mrp: units(p.mrp),
          available: !!((inv.inStock != null ? inv.inStock : it.inStock) && it.isAvail !== false && !(v.slotInfo && v.slotInfo.isAvail === false)),
          id,
        });
      }
    };
    const walk = (o, d) => {   // product groups live in GridWidget.gridElements.infoWithStyle.items and OOSItemCollectionCard.items.items
      if (!o || typeof o !== 'object' || d > 8) return;
      if (Array.isArray(o)) { for (const x of o) walk(x, d + 1); return; }
      if (Array.isArray(o.variations) && o.displayName) { take(o); return; }
      for (const k in o) if (k !== 'analytics' && k !== 'layout') walk(o[k], d + 1);
    };
    walk((j2.data || {}).cards || [], 0);
    if (j2.statusCode && j2.statusCode !== 0) out.error = 'search statusCode ' + j2.statusCode + ' ' + ((j2.data || {}).statusMessage || '');
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
