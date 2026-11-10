import { useCallback, useState } from "react";

const STORAGE_KEY = "modellore.apiKey";

function readStoredKey(): string | null {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * The API key for this page. It lives in memory, and in sessionStorage only if
 * the person asks to keep it for the tab; sessionStorage is cleared when the
 * tab closes. It is never written to localStorage.
 */
export function useApiKey() {
  const [apiKey, setApiKeyState] = useState<string | null>(readStoredKey);

  const setApiKey = useCallback((value: string | null, keepForTab = false) => {
    setApiKeyState(value);
    try {
      if (value && keepForTab) {
        window.sessionStorage.setItem(STORAGE_KEY, value);
      } else {
        window.sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Storage can be unavailable (private windows, blocked site data); memory is enough.
    }
  }, []);

  return { apiKey, setApiKey };
}
