(async () => {
  const lat = %(lat)s, lon = %(lon)s, q = %(q)s;
  const MAX_RES = 3, MAX_ITEMS = 15;
  let csrf = (document.cookie.match(/(?:^|; )csrf=([^;]+)/) || [])[1];
  if (!csrf) { try { csrf = (await (await fetch('/webroutes/auth/csrf', {credentials: 'include'})).json()).csrf; } catch (e) {} }
  const H = {'x-zomato-csrft': csrf || '', 'content-type': 'application/json'};
  const call = async (url, body) => {
    const r = await fetch(url, body === undefined ? {credentials: 'include', headers: H} : {method: 'POST', credentials: 'include', headers: H, body: JSON.stringify(body)});
    let j = null; try { j = await r.json(); } catch (e) {}
    return {status: r.status, j};
  };
  // 1) delivery point -> Zomato location (subzone/city ids, o2Serviceable); also becomes the session location the menu JSON reads
  const lp = new URLSearchParams({lat, lon, entity_id: '0', entity_type: '', userDefinedLatitude: lat, userDefinedLongitude: lon, placeId: '', placeType: '', placeName: '', cellId: '0', addressId: '0', isOrderLocation: '1', res_id: '0', pageType: 'search', persist: 'false'});
  let lg;
  try { lg = await call('/webroutes/location/get?' + lp.toString()); } catch (e) { return {status: 0, serviceable: null, error: 'location call failed: ' + e, products: []}; }
  const L = lg.j && lg.j.locationDetails;
  if (lg.status !== 200 || !L) return {status: lg.status, serviceable: null, error: 'no location answer', products: []};
  if (!L.o2Serviceable) return {status: lg.status, serviceable: false, location: L.displayTitle || L.cityName || null, products: []};
  const LK = ['addressId', 'entityId', 'entityType', 'locationType', 'isOrderLocation', 'cityId', 'latitude', 'longitude', 'userDefinedLatitude', 'userDefinedLongitude', 'entityName', 'orderLocationName', 'cityName', 'countryId', 'countryName', 'displayTitle', 'o2Serviceable', 'placeId', 'cellId', 'deliverySubzoneId', 'placeType', 'placeName', 'isO2City', 'fetchFromGoogle', 'fetchedFromCookie', 'isO2OnlyCity', 'addressBlocker'];
  // 2) the header search box's autosuggest for that location -> restaurants (serviceable, ETA, menu URL) and dish ids
  const sp = new URLSearchParams();
  for (const k of LK) sp.set(k, L[k] == null ? '' : String(L[k]));
  sp.set('q', q); sp.set('context', 'delivery');
  let as;
  try { as = await call('/webroutes/search/autoSuggest?' + sp.toString()); } catch (e) { return {status: 0, serviceable: true, error: 'search call failed: ' + e, products: []}; }
  if (as.status !== 200 || !as.j) return {status: as.status, serviceable: true, error: 'no search answer', products: []};
  const STOP = ['the', 'and', 'with', 'a', 'an', 'of', 'in', 'for', 'from', 'order', 'online', 'near', 'me'];
  const words = (s) => (String(s || '').toLowerCase().replace(/['’`]/g, '').match(/[a-z0-9]+/g) || []).filter(w => !STOP.includes(w));
  const qw = words(q);
  const results = as.j.results || [];
  const isRest = (r) => r && r.info && r.order && r.order.actionInfo && r.order.actionInfo.clickUrl;
  const canOrder = (r) => r.order.isServiceable && r.order.hasOnlineOrdering !== false;
  const rests = results.filter(r => r.entityType === 'restaurant' && isRest(r));
  const dishSugg = results.filter(r => r.entityType === 'universal_dish');
  const dishes = dishSugg.map(r => r.name);
  // brand words: query words in a suggested restaurant's name but in no dish suggestion ("dominos", not "pizza")
  const dishWords = new Set(dishes.flatMap(words));
  const brand = qw.filter(w => !dishWords.has(w) && rests.some(r => words(r.info.name).includes(w)));
  const named = (r) => brand.some(w => words(r.info.name).includes(w));
  const namedServ = rests.filter(r => named(r) && canOrder(r));
  const dw = qw.filter(w => !brand.includes(w));
  let pick = namedServ.slice(0, MAX_RES), via = 'named_restaurant';
  if (!pick.length && dishSugg.length) {
    // no named restaurant delivering here: restaurants serving the best-matching dish (the dish suggestion's listing)
    const score = (d) => words(d.name).filter(w => dw.includes(w)).length * 10 - words(d.name).length;
    const dish = dishSugg.slice().sort((a, b) => score(b) - score(a))[0];
    const applied = [{filterType: 'category_sheet', filterValue: 'delivery_home', isHidden: true, isApplied: true, postKey: '{"category_context":"delivery_home"}'},
      {filterType: 'universal_dish', filterValue: String(dish.entityId), isHidden: true, isApplied: true, postKey: JSON.stringify({universal_dish_ids: [String(dish.entityId)]})}];
    const body = {context: 'delivery', filters: JSON.stringify({searchMetadata: {}, dineoutAdsMetaData: {}, appliedFilter: applied, urlParamsForAds: {}})};
    for (const k of LK) body[k] = L[k];
    try {
      const af = await call('/webroutes/search/applyFilter', body);
      const sr = (af.j && af.j.pageData && af.j.pageData.sections && af.j.pageData.sections.SECTION_SEARCH_RESULT) || [];
      pick = sr.filter(r => isRest(r) && canOrder(r)).slice(0, MAX_RES);
      via = 'dish:' + dish.name;
    } catch (e) {}
  }
  if (!pick.length) { pick = rests.filter(canOrder).slice(0, MAX_RES); via = 'search'; }
  const base = {serviceable: true, location: L.displayTitle, named_restaurant_found: brand.length ? namedServ.length > 0 : null, via, dishes};
  if (!pick.length) return Object.assign({status: as.status, restaurants: [], products: []}, base);
  // 3) each picked restaurant's online-order menu (the JSON the restaurant's /order page loads) -> items matching the dish words
  const firstPath = (groups) => { const names = []; let g = groups; for (let d = 0; d < 4 && Array.isArray(g) && g.length; d++) { const gr = g[0].group || g[0]; const it = gr.items && gr.items[0] && (gr.items[0].item || gr.items[0]); if (!it) break; names.push(it.name); g = it.groups; } return names; };
  const sizesOf = (groups) => { let g = groups; for (let d = 0; d < 4 && Array.isArray(g) && g.length; d++) { const gr = g[0].group || g[0]; const its = (gr.items || []).map(x => x.item || x); if (/size|portion|quantity|serves/i.test(gr.name || '') && its.length) return its.map(x => ({name: x.name, price: x.price})); g = its[0] && its[0].groups; } return null; };
  const menus = await Promise.all(pick.map(async (r) => {
    try { const pg = await call('/webroutes/getPage?page_url=' + encodeURIComponent(r.order.actionInfo.clickUrl) + '&location=&isMobile=0'); return {r, status: pg.status, pd: pg.j && pg.j.page_data}; }
    catch (e) { return {r, status: 0, pd: null}; }
  }));
  const products = [], restaurants = [];
  for (const {r, status, pd} of menus) {
    const rname = r.info.name || '';
    const od = (pd && pd.orderDetails) || {};
    const eta = r.order.deliveryTime || od.deliveryTime || null;
    restaurants.push({name: rname, id: r.info.resId, url: r.order.actionInfo.clickUrl, eta, distance: r.distance || null, serviceable: !!od.isServiceable, menu_status: status});
    if (!pd || !pd.order || !pd.order.menuList) continue;
    const seen = new Set(), items = [];
    for (const m of pd.order.menuList.menus || []) for (const c of (m.menu && m.menu.categories) || []) for (const x of (c.category && c.category.items) || []) {
      const it = x.item; if (!it || !it.name || seen.has(it.id)) continue;
      seen.add(it.id);
      const iw = words(it.name);
      const hit = dw.length ? dw.filter(w => iw.some(v => v === w || (w.length > 3 && v.startsWith(w)))).length : 1;
      if (hit > 0) items.push({it, hit, menu: m.menu.name});
    }
    items.sort((a, b) => b.hit - a.hit || a.it.name.length - b.it.name.length);
    for (const {it, hit, menu} of items) {
      products.push({
        name: it.name, pack: firstPath(it.groups).join(', ') || null, price: it.min_price || it.display_price || it.price || null, mrp: null,
        available: it.item_state === 'available' && !!od.isServiceable && r.order.isDeliveringNow !== false,
        id: it.id, restaurant: rname, restaurant_id: r.info.resId, eta, distance: r.distance || null, menu,
        sizes: sizesOf(it.groups), veg: (it.dietary_slugs || []).includes('veg'), match: dw.length ? Math.round(hit / dw.length * 100) / 100 : null,
      });
    }
  }
  products.sort((a, b) => (b.match || 0) - (a.match || 0));
  return Object.assign({status: menus.some(m => m.status === 200) ? 200 : (menus[0] && menus[0].status) || 0, restaurants, products: products.slice(0, MAX_ITEMS)}, base);
})()
