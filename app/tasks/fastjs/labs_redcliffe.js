(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const V1 = 'https://app.redcliffelabs.com/api/v1/', V2 = 'https://app.redcliffelabs.com/api/v2/';
  // GET with a time limit; never throws. s = HTTP status (0 = network error / timeout).
  const get = async (u, ms) => {
    const c = new AbortController(); const t = setTimeout(() => c.abort(), ms || 10000);
    try { const r = await fetch(u, {signal: c.signal}); let j = null; try { j = await r.json(); } catch (e) {} return {s: r.status, j}; }
    catch (e) { return {s: 0, j: null}; } finally { clearTimeout(t); }
  };
  // Redcliffe prices by CITY. Map the address to a city and check home collection:
  // lat/lon -> geofence (what the site's "use my location" does), pincode -> list of served localities.
  const [geo, pr] = await Promise.all([
    lat && lon ? get(V1 + 'booking/checking-geofence-area-lat-long/?latitude=' + encodeURIComponent(lat) + '&longitude=' + encodeURIComponent(lon), 8000) : null,
    pin ? get(V1 + 'phlebo/pincode/?code=' + encodeURIComponent(pin), 8000) : null,
  ]);
  const geoOk = geo && geo.j && (geo.s === 200 || geo.s === 400) ? geo.j.status === true : null;
  const rows = pr && pr.s === 200 && pr.j && Array.isArray(pr.j.results) ? pr.j.results.filter((x) => x && String(x.pincode) === String(pin) && x.is_active !== false && x.city) : [];
  const pinOk = pr && (pr.s === 200 || pr.s === 400) ? rows.length > 0 : null;
  // the pincode list has stray rows in other cities (492001 also lists Chandigarh, Thane): take the most common city
  const cnt = {}; rows.forEach((x) => { cnt[x.city] = (cnt[x.city] || 0) + 1; });
  const pinCity = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0] || null;
  const city = (geoOk && geo.j.city_name) || pinCity;
  const serviceable = geoOk !== null ? geoOk : pinOk;
  if (!city) {
    if (serviceable === false) return {status: 200, serviceable: false, items: [], slots: null};
    const bad = [geo, pr].find((x) => x && x.s !== 200);
    return {status: bad ? bad.s : 400, serviceable: null, items: [], slots: null};
  }
  const url = (k, n) => V2 + 'package/get_packages_data_scored/?gamma=true&include_crm=true&page=1&search=' + encodeURIComponent(q) +
    '&city=' + encodeURIComponent(city) + '&package_or_test=' + k + '&limit=' + n + '&audio=false&source_type=web_consumer';
  const [ts, ps] = await Promise.all([get(url('test', 6)), get(url('package', 6))]);
  const okT = ts.s === 200 && ts.j && Array.isArray(ts.j.results), okP = ps.s === 200 && ps.j && Array.isArray(ps.j.results);
  if (!okT && !okP) return {status: ts.s !== 200 ? ts.s : (ps.s !== 200 ? ps.s : 0), serviceable, items: [], slots: null};
  const num = (v) => (v == null || v === '' || isNaN(+v) ? null : +v);
  const slugCity = city.toLowerCase().replace(/_/g, '-').replace(/ /g, '-');
  const list = [].concat(okT ? ts.j.results : [], okP ? ps.j.results : [])
    .filter((p) => p && p.name && (p.offer_price != null || p.package_city_prices))
    .sort((a, b) => (+b._score || 0) - (+a._score || 0)); // both lists use the same search scores
  const items = list.map((p) => {
    const cp = p.package_city_prices || {}; // the city's own price (what the site shows), else the default
    const kind = p.package_or_test === 'package' ? 'package' : 'test';
    return {
      name: p.name, kind,
      price: num(cp.offer_price != null ? cp.offer_price : p.offer_price),
      mrp: num(cp.package_price != null ? cp.package_price : p.package_price),
      tests_included: num(p.parameter),
      fasting: typeof p.is_fasting_required === 'boolean' ? p.is_fasting_required : null,
      report_time: cp.tat_time || p.tat_time || null,
      home_collection: typeof p.home_collection === 'boolean' ? p.home_collection : null,
      id: p.code || (p.id != null ? String(p.id) : null),
      url: p.slug ? 'https://redcliffelabs.com/' + slugCity + (kind === 'package' ? '/package/' : '/tests/') + p.slug : (p.web_link || null),
    };
  });
  // Collection slots need a signed-in user (booking_slot_collection_date answers 401 to guests): not read.
  return {status: 200, serviceable, items, slots: null};
})()
