# Build with AWS analytics

This course shares GA4 property **Build with AWS** (556133375), web stream
15855031162, and Measurement ID **G-YLB0CNYQ8Z** with marcelops.com, OCR Rush,
and `buildwithaws.substack.com`.

`site_src/build.py` reads `GA4_MEASUREMENT_ID` and adds
`site_src/assets/analytics.js` through the common page shell. New lessons and
pages generated through that shell inherit analytics. The deployment workflow
supplies the ID, with an optional Actions variable of the same name overriding
it. An empty ID disables injection; malformed nonempty IDs fail the build.
The runtime excludes localhost and preview domains.

The script is identical to the homepage and OCR copies. Keep the three copies
in sync. GA4 initializes once; its default config and Enhanced Measurement own
page views. Do not add another tag or manually track page_view.

`build_with_aws_click` tracks publication navigation with `page_path`,
`cta_id`, `cta_intent`, `link_domain`, and query-free `link_url`. Course CTA IDs
are `course_header_subscribe`, `course_footer`, and
`course_stage_notification`. Subscription intent is distinct from completed
subscription. Colab and other external links use enhanced outbound clicks.

Existing UTMs and Google's `_gl` decoration are preserved. Ordinary same-tab
publication navigation waits no more than 300 ms for event dispatch after
Google loads. The original click still bubbles to Google's linker listener;
the final decorated href is used for navigation.

Configure exact GA4 domain matches for `marcelops.com`, `www.marcelops.com`,
and `buildwithaws.substack.com`. Keep enhanced page views (including browser
history), scrolls and outbound clicks enabled. Register event-scoped custom
dimensions for `cta_id` and `cta_intent` to compare CTAs. Save the same
**G-YLB0CNYQ8Z** in Substack Settings → Analytics → Google Analytics Measurement
ID; no Substack application-code changes are needed.

Verify after publishing that the course sends one page view per page and one
publication event per click. Check `_gl` on the actual Substack destination
and matching GA request `cid` and `sid` within the same active session. Verify
Substack's real subscription event before treating it as a key event.
