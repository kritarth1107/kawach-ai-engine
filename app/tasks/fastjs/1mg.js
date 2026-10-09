(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const API = '/pwa-dweb-api';
  const getJson = async (u, h) => { const r = await fetch(API + u, {headers: h || {}}); let j = null; try { j = await r.json(); } catch (e) {} return {r, j}; };
  // 1mg prices and stock by CITY (guests cannot pick a pincode). Map the pincode to 1mg's city name, else use lat/lon.
  let city = null, status = 0;
  if (pin) { const {r, j} = await getJson('/api/v4/pincode/' + encodeURIComponent(pin)); status = r.status; city = j && j.data && j.data.city; }
  if (!city && lat && lon) {
    const {r, j} = await getJson('/location/latlng/' + encodeURIComponent(lat) + ',' + encodeURIComponent(lon));
    status = r.status; city = j && j.result && j.result[0] && j.result[0].city;
  }
  if (!city) return {status: status && status !== 200 ? status : 404, serviceable: null, eta: null, products: []};
  const H = {'x-city': city, accept: 'application/vnd.healthkartplus.v4+json'};
  const [sv, sr] = await Promise.all([
    getJson('/api/v4/city-serviceable?city=' + encodeURIComponent(city), H).catch(() => ({})),
    fetch(API + '/api/v4/search/all?q=' + encodeURIComponent(q) + '&city=' + encodeURIComponent(city) +
      '&filter=&page_number=0&scroll_id=&per_page=10&types=sku,allopathy&sort=relevance&fetch_eta=true&is_city_serviceable=true', {headers: H}),
  ]);
  if (!sr.ok) return {status: sr.status, serviceable: null, eta: null, products: []};
  const sj = await sr.json();
  const svd = sv && sv.j && sv.j.data;
  const num = (s) => { if (s == null) return null; const t = String(s).replace(/[₹,\s]/g, ''); return /^\d+(\.\d+)?$/.test(t) ? +t : s; };
  const strip = (s) => s ? String(s).replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim() : null;
  const etaOf = (p) => (p.ga_data && p.ga_data.info && p.ga_data.info.text) || strip(p.eta) || null;
  const list = ((sj.data && sj.data.search_results) || []).filter((p) => p && p.name && p.prices);
  const products = list.map((p) => ({
    name: p.name, pack: p.label || null,
    price: num(p.prices.discounted_price || p.prices.mrp), mrp: num(p.prices.mrp),
    available: !!p.available, rx_required: !!p.rx_required, id: p.id != null ? String(p.id) : null,
  }));
  const first = list.find((p) => p.available && etaOf(p));
  return {status: sr.status, serviceable: svd ? !!(svd.serviceable && svd.pharma_available !== false) : null,
    eta: first ? etaOf(first) : null, products};
})()
