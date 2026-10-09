(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const API = '/pwa-dweb-api';
  const enc = encodeURIComponent;
  const get = async (u, h, ms) => {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), ms || 10000);
    try {
      const r = await fetch(API + u, {headers: h || {}, signal: c.signal});
      let j = null; try { j = await r.json(); } catch (e) {}
      return {s: r.status, j};
    } catch (e) { return {s: 0, j: null}; } finally { clearTimeout(t); }
  };
  const out = (status, serviceable, items) => ({status, serviceable, items: items || [], slots: null});
  // 1mg Labs prices by CITY (as on the pharmacy side). Map the pincode to 1mg's city name, else use lat/lon.
  let city = null, status = 0;
  if (pin) { const r = await get('/api/v4/pincode/' + enc(pin)); status = r.s; city = r.j && r.j.data && r.j.data.city; }
  if (!city && lat && lon) {
    const r = await get('/location/latlng/' + enc(lat) + ',' + enc(lon));
    status = r.s; city = r.j && r.j.result && r.j.result[0] && r.j.result[0].city;
  }
  if (!city) return out(status && status !== 200 ? status : 404, null);
  const H = {'x-city': city, accept: 'application/vnd.healthkartplus.v4+json'};
  const sr = await get('/api/labs/v1/search/all?city=' + enc(city) + '&q=' + enc(q), H);
  const err = sr.j && sr.j.error && sr.j.error.errors && sr.j.error.errors[0];
  // Labs not offered in this city: 400 {"type":"city_not_serviceable"}. A definite answer, not a failure.
  if (sr.s === 400 && err && err.type === 'city_not_serviceable') return out(200, false);
  if (sr.s !== 200 || !sr.j || !sr.j.data) return out(sr.s, null);
  const num = (s) => { if (s == null) return null; const t = String(s).replace(/[₹,\s]/g, ''); return /^\d+(\.\d+)?$/.test(t) ? +t : null; };
  const strip = (s) => s ? String(s).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() : null;
  const list = (sr.j.data.search_results || []).filter((p) => p && p.id != null && p.name && p.prices);
  const items = list.map((p) => {
    const sub = strip(p.subheading && p.subheading.text) || '';
    const m = sub.match(/(\d+)\s+tests?/i);
    const cat = String(p.category || '').toLowerCase();
    return {
      name: p.name, kind: p.test_type === 'package' ? 'package' : 'test',
      price: num(p.prices.discounted_price) != null ? num(p.prices.discounted_price) : num(p.prices.mrp), mrp: num(p.prices.mrp),
      tests_included: m ? +m[1] : null, fasting: null,
      report_time: strip(p.eta && p.eta.text),
      home_collection: cat === 'pathology' ? true : (cat === 'radiology' ? false : null),
      id: String(p.id), url: 'https://www.1mg.com' + (p.url || ('/labs/test/' + p.id)),
    };
  });
  // Fasting is only on the test page: read "Preparations" from its data for the first 3 results (in parallel).
  // The full list is in the bottom sheet {"header":"Preparations","items":[{"header":"Overnight fasting (8-12 hrs) is
  // required..."},...]}; a short line (sub_header) sits next to the "Preparations" heading on some pages.
  const prepOf = (j) => {
    let full = null, short = null;
    const walk = (o) => {
      if (full || !o || typeof o !== 'object') return;
      if (!Array.isArray(o) && typeof o.header === 'string' && strip(o.header) === 'Preparations') {
        if (Array.isArray(o.items) && o.items.length) { full = o.items.map((x) => (x && x.header) || '').join(' '); return; }
        if (!short && typeof o.sub_header === 'string') short = o.sub_header;
      }
      for (const k in o) walk(o[k]);
    };
    walk(j);
    return full || short;
  };
  const fastingOf = (t) => {
    if (!t) return null;
    if (/\b(no|not)\s+(special\s+)?(preparation|fasting)|fasting\s+(is\s+)?not\s+(required|needed)|not\s+mandatory|non[\s-]?fasting/i.test(t)) return false;
    return /fast/i.test(t) ? true : null;
  };
  await Promise.all(items.slice(0, 3).map(async (it) => {
    const r = await get('/api/labs/v1/test/' + enc(it.id) + '/static?city=' + enc(city), H, 5000);
    if (r.s === 200 && r.j) it.fasting = fastingOf(strip(prepOf(r.j.data)));
  }));
  return out(200, true, items);
})()
