/* Build with AWS GA4. Copied unchanged to the independently deployed project sites. */
(() => {
  'use strict';
  const measurementId = document.currentScript?.dataset.measurementId || '';
  const productionHosts = ['marcelops.com', 'www.marcelops.com'];
  if (!/^G-[A-Z0-9]+$/.test(measurementId) ||
      !productionHosts.includes(location.hostname) ||
      window.__buildWithAwsAnalytics) return;
  window.__buildWithAwsAnalytics = measurementId;

  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
  window.gtag('js', new Date());
  // GA4 sends page-load views. Shared history tracking is off: Substack owns its routes.
  // Do not also subscribe to MkDocs location$ or send manual page_view events.
  window.gtag('config', measurementId);
  let googleReady = false;
  const tag = document.createElement('script');
  tag.onload = () => { googleReady = true; };
  tag.async = true;
  tag.src = 'https://www.googletagmanager.com/gtag/js?id=' + measurementId;
  document.head.appendChild(tag);

  const newsletterHost = 'buildwithaws.substack.com';
  function placement(element) {
    const explicit = element.dataset.analyticsId || element.id;
    if (/^[a-zA-Z0-9_-]{1,64}$/.test(explicit || '')) return explicit;
    for (const area of ['header', 'footer', 'nav', 'main', 'article']) {
      if (element.closest(area)) return area + '_link';
    }
    return 'content_link';
  }
  function trackClick(event) {
    if (event.defaultPrevented || (event.type === 'click' && event.button !== 0) ||
        (event.type === 'auxclick' && event.button !== 1)) return;
    const element = event.target instanceof Element
      ? event.target.closest('a[href],button[data-analytics-event]') : null;
    if (!element) return;
    const parameters = {
      send_to: measurementId,
      page_path: location.pathname,
      cta_id: placement(element)
    };
    if (element.matches('button[data-analytics-event="newsletter_cta_click"]')) {
      if (!element.disabled) window.gtag('event', 'newsletter_cta_click', {
        ...parameters, cta_intent: 'subscribe', destination: 'substack_embed'
      });
      return;
    }
    if (!element.matches('a[href]') || element.hasAttribute('download')) return;
    let destination;
    try { destination = new URL(element.href, location.href); } catch { return; }
    if (destination.hostname !== newsletterHost ||
        !['https:', 'http:'].includes(destination.protocol)) return;
    // Give the Google event a bounded chance to finish before replacing this page.
    // Let the click keep bubbling so Google's standard linker can decorate href.
    if (googleReady && event.type === 'click' && !event.metaKey && !event.ctrlKey &&
        !event.shiftKey && !event.altKey && (!element.target || element.target === '_self')) {
      event.preventDefault();
      let followed = false;
      const follow = () => {
        if (followed) return;
        followed = true;
        location.assign(element.href); // Read Google's decorated URL after propagation.
      };
      parameters.event_callback = () => setTimeout(follow, 0);
      const navigationTimeout = 1000;
      parameters.event_timeout = navigationTimeout;
      setTimeout(follow, navigationTimeout); // Navigation still works if analytics is blocked/fails.
    }
    // Cross-domain destinations are excluded from GA4's outbound `click` event.
    // This is navigation/intent, never evidence of a successful subscription.
    window.gtag('event', 'build_with_aws_click', {
      ...parameters,
      cta_intent: element.dataset.analyticsIntent === 'subscribe' ||
        /^\/subscribe(?:\/|$)/.test(destination.pathname) ? 'subscribe' : 'read',
      link_domain: newsletterHost,
      link_url: destination.origin + destination.pathname
    });
    // Never rewrite href, UTM parameters, _gl, or stop event propagation.
    // Do not collect link text, form contents, email addresses, or URL query values.
  }
  document.addEventListener('click', trackClick);
  document.addEventListener('auxclick', trackClick);
})();
