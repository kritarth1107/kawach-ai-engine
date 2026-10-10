(async () => {
  // Inside Zomato's payment frame on Blinkit's checkout: make Cash the open option. Only a yes/no comes back.
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  for (let i = 0; i < 80 && !/\bCash\b/.test(document.body ? document.body.innerText : ''); i++) await sleep(100);
  const open = () => /keep exact change/i.test(document.body.innerText);
  if (!open()) { const c = [...document.querySelectorAll('div, span, p, button')].find(e => (e.innerText || '').trim() === 'Cash'); if (c) { c.click(); await sleep(900); } }
  return { cash: open(), seen: /\bCash\b/.test(document.body.innerText) };
})()
