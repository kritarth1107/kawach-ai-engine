(async () => {
  const q = %(q)s, pin = %(pincode)s;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const cookie = (n) => { const m = document.cookie.match(new RegExp('(?:^|; )' + n + '=([^;]*)')); return m ? decodeURIComponent(m[1]) : null; };
  const H = {'x-phone-platform': 'web', 'x-vendor': 'pharmeasy', accept: 'application/json'};
  const call = async (url, opts) => {
    const ac = new AbortController(); const t = setTimeout(() => ac.abort(), 10000);
    try {
      const r = await fetch(url, Object.assign({signal: ac.signal}, opts || {}));
      let j = null; try { j = await r.json(); } catch (e) {}
      return {status: r.status, j};
    } catch (e) { return {status: 0, j: null}; } finally { clearTimeout(t); }
  };
  // Lab-test serviceability of the pincode (the address form makes this call). Runs while the page starts up.
  const svcP = pin ? call('/diag-pwa/backend-api/diagnostics/catalog/v1/serviceability/pincodes',
    {method: 'POST', headers: Object.assign({'content-type': 'application/json'}, H), body: JSON.stringify({pincodes: [+pin || 0]})}) : null;
  // Let the page's own start-up finish: it writes its default X-Pincode cookie (Mumbai 400086) and looks it up.
  // A pincode written before that gets overwritten.
  for (let i = 0; i < 40; i++) {
    if (cookie('X-Pincode') && performance.getEntriesByType('resource').some((e) => e.name.includes('/api/app/fetchPincodeDetails'))) break;
    await sleep(150);
  }
  // Same as "Select Pincode -> Check" in the UI: only X-Pincode changes. X-Default-City stays as the UI keeps it (1);
  // with X-Default-City=867 (Raipur, as the pharmacy snippet writes) CloudFront answers the search with 403.
  const setPin = () => {
    if (!pin) return;
    document.cookie = 'X-Pincode=' + encodeURIComponent(pin) + '; path=/; max-age=31536000';
    if (cookie('X-Default-City') !== '1') document.cookie = 'X-Default-City=1; path=/; max-age=31536000';
  };
  const search = async () => {
    setPin();
    return call('/api/diagnostics/getAllSearchResults?q=' + encodeURIComponent(q) + '&page=1', {headers: H});
  };
  let sr = await search();
  if (pin && cookie('X-Pincode') !== pin) sr = await search(); // the page reset it mid-way: once more
  const svc = svcP ? await svcP : null;
  const sv = svc && svc.j && Array.isArray(svc.j.data) && svc.j.data[0];
  let serviceable = sv && typeof sv.is_serviceable === 'boolean' ? sv.is_serviceable : null;
  const list = (sr.j && sr.j.data && Array.isArray(sr.j.data.search_data)) ? sr.j.data.search_data : null;
  if (sr.status !== 200 || !list) return {status: sr.status || 0, serviceable, items: [], slots: null};
  const num = (s) => (s == null || s === '' || isNaN(+s) ? null : +s);
  const fastingOf = (p) => {
    const f = (p.test_requirements || []).find((x) => /fasting/i.test(x.name || ''));
    if (!f || !f.value) return null;
    return !/not\s*required|^no$|not\s*needed/i.test(f.value);
  };
  const countOf = (p) => {
    const m = /(\d+)\s*tests?/i.exec(p.sub_text || '');
    if (m) return +m[1];
    if (p.item_type === 'test') return 1;
    const ti = p.tests_included;
    if (!ti) return null;
    const n = (ti.tests || []).length + (ti.profiles || []).reduce((a, x) => a + ((x.tests || []).length || 1), 0);
    return n || null;
  };
  const kept = list.filter((p) => p && p.item_name && p.item_id != null && (p.pe_selling_price || p.mrp));
  if (serviceable === null && kept.length) serviceable = kept.some((p) => p.is_serviceable_at_selected_pincode !== false);
  const shown = serviceable === false ? [] : kept.filter((p) => p.is_serviceable_at_selected_pincode !== false);
  const page = {test: 'tests', package: 'packages', profile: 'profile'}; // canonical product pages (/package/ redirects)
  const items = shown.map((p) => ({
    name: String(p.item_name).trim(), kind: p.item_type === 'package' ? 'package' : 'test',
    price: num(p.pe_selling_price) != null ? num(p.pe_selling_price) : num(p.starting_price != null ? p.starting_price : p.mrp),
    mrp: num(p.mrp), tests_included: countOf(p), fasting: fastingOf(p),
    report_time: (p.tat_details && p.tat_details.text) || null,
    home_collection: p.category === 'pathology' ? true : (p.category === 'radiology' ? false : null),
    id: p.item_type + ':' + p.item_id,
    url: location.origin + '/diagnostics/' + (page[p.item_type] || p.item_type) + '/' + p.slug,
  }));
  // Earliest home-collection slots, as the test page shows them ("EXPRESS SLOT Available at ..."): the page reads the
  // lab and collection centre of the item, then that lab's slot list for the pincode. Guest, read-only.
  let slots = null;
  const first = shown[0];
  if (first && pin && serviceable !== false && first.category === 'pathology') {
    const city = cookie('X-Default-City') || '1';
    const PH = Object.assign({'x-pincode': pin, 'x-default-city': city}, H);
    const pdp = await call('/diag-pwa/backend-api/diagnostics/catalog/v1/pdp/' + first.item_type + '/' + first.item_id, {headers: PH});
    const lab = pdp.j && pdp.j.data && (pdp.j.data.available_at || []).find((l) => l && l.item_type === 'lab' && l.is_active !== false);
    if (lab) {
      const sl = await call('/diag-pwa/diagnostic-api/api/slot/v2/slot-service/slot-list/address-city/' + encodeURIComponent(city) +
        '?lab_id=' + lab.item_id + '&mce_id=' + (lab.mce_id || 0) + '&address_pincode=' + encodeURIComponent(pin) + '&patient_quantity=1', {headers: H});
      const fd = sl.j && sl.j.data && sl.j.data.first_date;
      if (Array.isArray(fd)) {
        slots = fd.filter((s) => s && s.date && s.start_time && (s.available_capacity == null || s.available_capacity > 0)).slice(0, 3).map((s) =>
          s.date + ' ' + String(s.start_time).slice(0, 5) + '-' + String(s.end_time || '').slice(0, 5) + (s.express_slot ? ' express' : '') + (s.charge > 0 ? ' (extra Rs ' + s.charge + ')' : ''));
      }
    }
  }
  return {status: sr.status, serviceable, items, slots};
})()
