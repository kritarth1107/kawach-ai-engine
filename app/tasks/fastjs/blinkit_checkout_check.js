(async () => {
  // Blinkit checkout page (https://blinkit.com/checkout), main document only: the cart lines and address must match what the
  // family confirmed. The payment options live in Zomato's payment frame (zomato.com/zpaykit), checked separately.
  const want = JSON.parse(%(want)s);   // {count, prices: [..], address}
  const out = { ok: false };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const payNow = () => [...document.querySelectorAll('div, button, a')].find(e => (e.innerText || '').trim() === 'Pay Now' && e.offsetParent !== null);
  for (let i = 0; i < 200 && !payNow(); i++) await sleep(100);
  if (!payNow()) { out.problem = 'checkout page not ready (no Pay Now)'; return out; }
  const main = document.body.innerText.replace(/\s+/g, ' ');
  out.page = main.slice(0, 240);
  if (!new RegExp('\\b' + Number(want.count) + ' items?\\b').test(main)) { out.problem = 'item count differs'; return out; }
  for (const p of want.prices || []) if (!main.includes('₹' + p)) { out.problem = 'a price differs (₹' + p + ' not on the page)'; return out; }
  if (want.address && !main.toLowerCase().includes(String(want.address).toLowerCase().slice(0, 14))) { out.problem = 'address differs'; return out; }
  out.ok = true;
  return out;
})()
