(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const enc = encodeURIComponent;
  const cookie = (n) => { const m = document.cookie.match(new RegExp('(?:^|; )' + n + '=([^;]*)')); return m ? decodeURIComponent(m[1]) : null; };
  const call = async (u, opt, ms) => {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), ms || 10000);
    try {
      const r = await fetch(u, Object.assign({signal: c.signal}, opt || {}));
      let j = null; try { j = await r.json(); } catch (e) {}
      return {s: r.status, j};
    } catch (e) { return {s: 0, j: null}; } finally { clearTimeout(t); }
  };
  const out = (status, serviceable, items) => ({status, serviceable, items: items || [], slots: null});
  // AWS WAF first serves a challenge page that reloads into the real one. The real page (Laravel) carries the CSRF
  // token that the search calls need.
  let token = null;
  for (let i = 0; i < 40 && !token; i++) {
    const m = document.querySelector('meta[name="csrf-token"]');
    token = m && m.content;
    if (!token) await sleep(200);
  }
  if (!token) return out(0, null);
  const XH = {'x-requested-with': 'XMLHttpRequest'};
  // Serviceability and city: the same call the page makes with the browser's location. Healthians has no pincode
  // lookup for guests (the location box goes through Google Places), so lat/lon decide.
  let city = null, serviceable = null;
  if (lat && lon) {
    const r = await call('/getLocalityID?lat=' + enc(lat) + '&long=' + enc(lon), {headers: XH});
    if (r.j && r.j.status === true && r.j.data && r.j.data.city_name) { serviceable = true; city = r.j.data.city_name; }
    else if (r.j && r.j.status === false) return out(200, false); // "Out of Service Area"
  }
  if (!city) city = cookie('sNewLocation') || cookie('sLocation') || 'delhi';
  city = String(city).toLowerCase().replace(/ /g, '_');
  const form = (o) => { const b = new URLSearchParams(); for (const k in o) b.set(k, o[k]); return b.toString(); };
  const FH = Object.assign({'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'x-csrf-token': token}, XH);
  // 1) the search box's suggestions: names, ids and list prices (no offer price)
  const sr = await call('/packageSuggestionUrl', {method: 'POST', headers: FH,
    body: form({term: q, source: 'web', channel_user: 0, channel_type: 0})});
  if (sr.s !== 200 || !sr.j) return out(sr.s, serviceable);
  const PATH = ['pathology', 'genetic', 'genetic_pathology'];
  const sug = (Array.isArray(sr.j.response) ? sr.j.response : []).filter((p) => p && p.id && p.text && PATH.includes(p.product_type));
  // 2) offer price, fasting, report time: what the results page shows after picking one suggestion (one call each)
  const top = sug.slice(0, 4);
  let dstatus = 0;
  const det = await Promise.all(top.map(async (p) => {
    const r = await call('/' + enc(city) + '/ajaxOpenSearchPathology/pathology', {method: 'POST', headers: FH,
      body: form({'search_val[0][id]': p.id, 'search_val[0][text]': p.text, 'search_val[0][type]': p.product_type,
        _token: token, ChannelPartnerType: 0, ChannelPartnerId: 0})}, 8000);
    if (r.s !== 200 || !dstatus) dstatus = r.s;
    const ex = r.j && r.j.search_lists && r.j.search_lists.exact;
    return (Array.isArray(ex) && ex.find((x) => x && x.id === p.id)) || null;
  }));
  // suggestions found but no offer price for any of them (e.g. 419 = CSRF/session expired): not a usable answer
  if (top.length && !det.some(Boolean)) return out(dstatus && dstatus !== 200 ? dstatus : 0, serviceable);
  const num = (v) => (v == null || v === '' || isNaN(+v) ? null : +v);
  const items = top.map((p, i) => {
    const d = det[i];
    const type = String(p.id).split('_')[0]; // package | profile | parameter
    const sd = d && d.search_data && d.search_data[0];
    const slug = (sd && sd.link_rewrite) || p.link_rewrite;
    const ft = d ? num(d.fasting_time) : null;
    return {
      name: (d && d.name) || p.text, kind: type === 'package' ? 'package' : 'test',
      price: d ? num(d.discount_price) : null, mrp: num(d ? d.mrp : p.mrp),
      tests_included: num(d ? d.parameter_count : p.parameter_count),
      fasting: ft == null ? null : ft > 0,
      report_time: (d && (d.reporting_tat_display || d.reportTatTime)) || null,
      home_collection: true, id: String(p.id),
      url: slug ? 'https://www.healthians.com/' + type + '/' + city + '/' + slug : null,
    };
  }).filter((it) => it.price != null);
  return out(200, serviceable, items);
})()
