(async () => {
  /* Rapido guest fare estimate. Run in a tab on https://m.rapido.bike (any page of the PWA).
     The fareEstimate endpoint needs "thiqa" signature headers (x-km, x-m, x-ekref, x-pm) that the PWA's
     own Angular HTTP interceptor computes in WASM. So the call goes through the app's own HttpClient
     (found via app-root.__ngContext__); the services list (names) is a plain fetch. */
  const plat = Number(%(plat)s), plon = Number(%(plon)s), dlat = Number(%(dlat)s), dlon = Number(%(dlon)s);
  const pickup = String(%(pickup)s || 'Pickup'), drop = String(%(drop)s || 'Drop');
  const out = {status: 0, login_required: false, options: []};
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const findHttp = () => {
    const root = document.querySelector('app-root');
    if (!root || !root.__ngContext__) return null;
    const seen = new Set(), q = [[root.__ngContext__, 0]];
    while (q.length) {
      const [o, d] = q.shift();
      if (!o || typeof o !== 'object' || seen.has(o) || o instanceof Node || o === window) continue;
      seen.add(o);
      const h = o.httpClient;
      if (h && typeof h.post === 'function' && h.handler) return h;
      if (d > 6) continue;
      for (const k of (Array.isArray(o) ? o.keys() : Object.keys(o))) {
        let v; try { v = o[k]; } catch (e) { continue; }
        if (v && typeof v === 'object') q.push([v, d + 1]);
      }
    }
    return null;
  };
  /* minimal protobuf reader for the fareEstimate reply (a JSON array of bytes) */
  const pb = (b) => {
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength), f = [];
    let i = 0;
    const vi = () => { let r = 0, m = 1, x; do { x = b[i++]; r += (x & 127) * m; m *= 128; } while (x > 127); return r; };
    while (i < b.length) {
      const k = vi(), no = Math.floor(k / 8), t = k & 7;
      let v;
      if (t === 0) v = vi();
      else if (t === 1) { v = dv.getFloat64(i, true); i += 8; }
      else if (t === 2) { const l = vi(); v = b.subarray(i, i + l); i += l; }
      else if (t === 5) { v = dv.getFloat32(i, true); i += 4; }
      else break;
      f.push([no, t, v]);
    }
    return f;
  };
  const txt = (u8) => new TextDecoder().decode(u8);
  try {
    let http = findHttp();
    for (let n = 0; !http && n < 60; n++) { await sleep(200); http = findHttp(); }
    if (!http) { out.error = 'Rapido app HttpClient not found (page not bootstrapped?)'; return out; }
    const H = {accept: 'application/json, text/plain, */*', appid: '2', appversion: '214', authorization: 'Bearer',
      'channel-entity': 'customer', 'channel-host': 'browser', 'channel-name': 'pwa', 'content-type': 'application/json', version: '1.0'};
    const svP = fetch('/pwa/api/unup/location/services', {method: 'POST', headers: H, body: JSON.stringify({lat: plat, lng: plon})})
      .then((r) => r.json()).catch(() => null);
    const body = {pickupLocation: {lat: plat, lng: plon, displayName: pickup, address: pickup},
      dropLocation: {lat: dlat, lng: dlon, displayName: drop, address: drop}, deviceId: 'seo-route-pages'};
    const fe = await new Promise((resolve) => {
      const tm = setTimeout(() => resolve({status: 0, body: null, err: 'timeout'}), 15000);
      http.post('/pwa/api/unup/scc/fareEstimate', body, {observe: 'response'}).subscribe(
        (r) => { clearTimeout(tm); resolve({status: r.status, body: r.body}); },
        (e) => { clearTimeout(tm); resolve({status: e.status || 0, body: e.error, err: e.message}); });
    });
    out.status = fe.status;
    if (fe.status === 401 || fe.status === 403) { out.login_required = true; out.error = JSON.stringify(fe.body).slice(0, 200); return out; }
    if (!Array.isArray(fe.body)) { out.error = 'unexpected fareEstimate reply: ' + JSON.stringify(fe.body).slice(0, 200); return out; }
    const top = pb(Uint8Array.from(fe.body));
    const data = top.find((x) => x[0] === 2 && x[1] === 2);
    const quotes = data ? pb(data[2]).filter((x) => x[0] === 4 && x[1] === 2).map((x) => {
      const q = {};
      for (const [no, t, v] of pb(x[2])) q[no] = t === 2 ? txt(v) : v;
      return {parentServiceId: q[1], serviceId: q[2], min: q[3], max: q[4]};
    }) : [];
    const sv = await svP;
    const services = (sv && sv.data && Array.isArray(sv.data.data)) ? sv.data.data : [];
    const rupee = (n) => '₹' + Math.round(n);
    if (services.length) {
      for (const s of services) {
        const q = quotes.find((x) => x.serviceId === s._id) || quotes.find((x) => x.parentServiceId === s.parentServiceId);
        if (q) out.options.push({type: s.displayName, fare: q.min === q.max ? rupee(q.min) : rupee(q.min) + ' - ' + rupee(q.max), eta: null});
      }
    } else {
      for (const q of quotes) out.options.push({type: 'service ' + q.parentServiceId, fare: rupee(q.min) + ' - ' + rupee(q.max), eta: null});
    }
    if (!out.options.length) out.error = 'no fare quotes returned';
  } catch (e) {
    out.error = String(e && e.message || e).slice(0, 300);
  }
  return out;
})()
