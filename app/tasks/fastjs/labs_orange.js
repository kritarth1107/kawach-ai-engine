(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const GW = 'https://gateway-api.orangehealth.in/';
  // GET with a time limit; never throws. s = HTTP status (0 = network error / timeout).
  const get = async (u, ms) => {
    const c = new AbortController(); const t = setTimeout(() => c.abort(), ms || 10000);
    try { const r = await fetch(u, {signal: c.signal}); let j = null; try { j = await r.json(); } catch (e) {} return {s: r.status, j}; }
    catch (e) { return {s: 0, j: null}; } finally { clearTimeout(t); }
  };
  // Orange Health works in a few metros only. Its own codes (from the site's JS; note DEL = Gurugram, NDM = Delhi):
  const SLUG = {BLR: 'bangalore', MUM: 'mumbai', NDM: 'delhi', HYD: 'hyderabad', NOA: 'noida', DEL: 'gurgaon', PNQ: 'pune'};
  const BY_NAME = {bengaluru: 'BLR', bangalore: 'BLR', mumbai: 'MUM', delhi: 'NDM', 'new delhi': 'NDM', hyderabad: 'HYD', noida: 'NOA', gurugram: 'DEL', gurgaon: 'DEL', pune: 'PNQ'};
  let serviceable = null, code = null;
  if (lat && lon) {
    // Same check as the site's "Book now" location step: 200 = a collection hub serves this point, >300 = not served.
    const sv = await get(GW + 'health-api/api/v1/order/location/serviceable/?latitude=' + encodeURIComponent(lat) + '&longitude=' + encodeURIComponent(lon), 8000);
    if (sv.s === 200 && sv.j) {
      serviceable = true;
      code = SLUG[sv.j.city_code] ? sv.j.city_code : (BY_NAME[String(sv.j.city || '').toLowerCase()] || null);
    } else if (sv.s > 300) serviceable = false;
  }
  if (!code && serviceable !== false && pin) {
    // No usable lat/lon answer: pick the metro from the pincode (city level only, address not checked).
    const p3 = String(pin).slice(0, 3);
    const PIN = {'560': 'BLR', '562': 'BLR', '400': 'MUM', '401': 'MUM', '410': 'MUM', '110': 'NDM', '500': 'HYD', '501': 'HYD', '502': 'HYD', '201': 'NOA', '122': 'DEL', '411': 'PNQ', '412': 'PNQ'};
    code = PIN[p3] || null;
    if (!code && serviceable === null) serviceable = false; // outside every metro Orange Health works in
  }
  if (serviceable === false) return {status: 200, serviceable: false, items: [], slots: null};
  if (!code) return {status: 0, serviceable: null, items: [], slots: null}; // could not check the location
  const sr = await get(GW + 'cerebro-api/api/v3/search/aggregated?city_code=' + code + '&limit=8&search_substring=' + encodeURIComponent(q));
  if (sr.s !== 200 || !sr.j || typeof sr.j !== 'object') return {status: sr.s || 0, serviceable, items: [], slots: null};
  const num = (v) => (v == null || v === '' || isNaN(+v) ? null : +v);
  const cnt = (v) => (num(v) > 0 ? num(v) : null); // a few tests carry numberOfParameters 0
  const tat = (x) => {
    const s = (x.appTatInformation && x.appTatInformation.tat_string) || x.labTatString || null;
    return s ? (/^\d/.test(s) ? 'Within ' + s : 'By ' + s) : null; // "6 hours" -> "Within 6 hours"; "Fri, 16 Oct" -> "By Fri, 16 Oct"
  };
  const site = 'https://www.orangehealth.in';
  const testPath = code === 'BLR' ? '/lab-test-bangalore/' : '/lab-tests-' + SLUG[code] + '/';
  const tests = [].concat(sr.j.tests || [], sr.j.panels || []).filter((t) => t && (t.testName || t.panelName || t.name) && t.consumerPrice != null).map((t) => ({
    name: String(t.testName || t.panelName || t.name).trim(), kind: 'test',
    price: num(t.consumerPrice), mrp: num(t.strikeOffPrice != null ? t.strikeOffPrice : t.consumerPrice),
    tests_included: cnt(t.numberOfParameters), fasting: t.isFastingRequired == null ? null : +t.isFastingRequired === 1,
    report_time: tat(t), home_collection: null,
    id: String(t.orangeTestId != null ? t.orangeTestId : t.id),
    url: t.slug && t.hasProductPage !== false ? site + testPath + t.slug : null,
  }));
  const pkgs = (sr.j.packages || []).filter((p) => p && p.packageName && p.consumerPrice != null).map((p) => ({
    name: String(p.packageName).trim(), kind: 'package',
    price: num(p.consumerPrice), mrp: num(p.totalTestsPrice != null ? p.totalTestsPrice : p.consumerPrice), // the struck-out price the site shows
    tests_included: cnt(p.numberOfParameters), fasting: p.isFastingRequired == null ? null : +p.isFastingRequired === 1,
    report_time: tat(p), home_collection: null,
    id: String(p.orangePackageId != null ? p.orangePackageId : p.id),
    url: p.slug && p.hasProductPage !== false ? site + '/health-checkups-' + SLUG[code] + '/' + p.slug : null,
  }));
  // The search gives no scores (searchScore is null): single tests first, then checkups that contain the search term.
  // Collection slots are only offered after phone/OTP sign-in at checkout: not read.
  return {status: 200, serviceable, items: tests.concat(pkgs), slots: null};
})()
