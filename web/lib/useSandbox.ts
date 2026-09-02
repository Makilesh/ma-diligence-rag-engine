"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { createDeal, ingestDocument, purgeDeal } from "./api";
import type { Deal } from "./types";

/** Where the active sandbox deal id is parked so a reload can still clean it up. */
const STORAGE_KEY = "redline.sandbox.deal";

/**
 * Manages the lifetime of a visitor's temporary deal.
 *
 * Uploaded documents must not outlive the session that created them, which is
 * arranged on three triggers because none of them is individually reliable:
 *
 *   1. `pagehide` fires a beacon. This is the one that makes closing the tab
 *      feel like deletion, and it covers navigation, reload and tab close. It
 *      is explicitly best-effort in the spec.
 *   2. A leftover id in sessionStorage is purged on next mount. This catches
 *      the reload case where the beacon was dropped.
 *   3. The server's TTL sweeper purges anything still standing. This is the
 *      only guarantee, and it is what covers a crashed tab, a killed mobile
 *      browser, or a machine that simply loses power.
 *
 * `beforeunload` is deliberately not used: it is unreliable on mobile, where
 * browsers background and kill tabs without ever firing it. `pagehide` fires in
 * both cases.
 */
export function useSandbox(
  onDealCreated: (deal: Deal) => void,
  onDealDiscarded: () => void,
) {
  const [sandboxDeal, setSandboxDeal] = useState<Deal | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Read in the handler rather than closing over state, so the unload path
  // always sees the current deal instead of the one captured at mount.
  const dealRef = useRef<Deal | null>(null);
  dealRef.current = sandboxDeal;

  // Trigger 2: clean up whatever the previous page left behind.
  useEffect(() => {
    const orphan = sessionStorage.getItem(STORAGE_KEY);
    if (orphan) {
      purgeDeal(orphan);
      sessionStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Trigger 1: the beacon.
  useEffect(() => {
    const onPageHide = () => {
      const deal = dealRef.current;
      if (deal) purgeDeal(deal.deal_id, true);
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        let deal = dealRef.current;

        if (!deal) {
          deal = await createDeal(
            `Sandbox · ${new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}`,
            true,
          );
          sessionStorage.setItem(STORAGE_KEY, deal.deal_id);
          setSandboxDeal(deal);
          dealRef.current = deal;
          onDealCreated(deal);
        }

        await ingestDocument(deal.deal_id, file);
        return deal;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
        return null;
      } finally {
        setUploading(false);
      }
    },
    [onDealCreated],
  );

  /** Deletes the sandbox now, at the visitor's request. */
  const discard = useCallback(() => {
    const deal = dealRef.current;
    if (!deal) return;
    purgeDeal(deal.deal_id);
    sessionStorage.removeItem(STORAGE_KEY);
    setSandboxDeal(null);
    dealRef.current = null;
    // The caller has to re-read the deal list: the purged deal is still in it,
    // and if it was the selected one, every subsequent query would be scoped to
    // a deal that no longer exists and would refuse for lack of context.
    onDealDiscarded();
  }, [onDealDiscarded]);

  return { sandboxDeal, uploading, error, upload, discard };
}
