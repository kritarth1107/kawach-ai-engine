(() => {
  // Press Pay Now (cash was checked in the payment frame just before). The app then loads the order page with a full
  // navigation, so this returns at once and the caller watches the tab's address for /track/<cart>/<order>.
  const btn = [...document.querySelectorAll('div, button, a')].filter(e => (e.innerText || '').trim() === 'Pay Now' && e.offsetParent !== null).pop();
  if (!btn) return { clicked: false, problem: 'no Pay Now' };
  setTimeout(() => btn.click(), 30);
  return { clicked: true };
})()
