import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export interface I18nProviderProps {
  children: ReactNode;
  initialLocale?: string;
  fallbackLocale?: string;
  messages?: Record<string, Record<string, string>>;
}

export interface I18nContextValue {
  locale: string;
  fallbackLocale: string;
  messages: Record<string, Record<string, string>>;
  setLocale: (nextLocale: string) => void;
  loadMessages: (locale: string, bundle: Record<string, string>) => void;
  translate: (key: string, vars?: Record<string, unknown>) => string;
  t: (key: string, vars?: Record<string, unknown>) => string;
  has: (key: string) => boolean;
}

const DEFAULT_LOCALE = "en";

const I18nContext = createContext<I18nContextValue | undefined>(undefined);

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\]/g, "\$&");
}

function formatTemplate(template: string, vars?: Record<string, unknown>): string {
  if (!vars) {
    return template;
  }
  return Object.entries(vars).reduce((output, [key, raw]) => {
    const pattern = new RegExp(`{{\s*${escapeRegExp(key)}\s*}}`, "g");
    return output.replace(pattern, String(raw));
  }, template);
}

function resolveMessage(
  key: string,
  locale: string,
  fallbackLocale: string,
  registry: Record<string, Record<string, string>>,
): string {
  const localeBundle = registry[locale] ?? {};
  if (key in localeBundle) {
    return localeBundle[key];
  }
  const fallbackBundle = registry[fallbackLocale] ?? {};
  if (key in fallbackBundle) {
    return fallbackBundle[key];
  }
  return key;
}

export function I18nProvider({
  children,
  initialLocale = DEFAULT_LOCALE,
  fallbackLocale = DEFAULT_LOCALE,
  messages = {},
}: I18nProviderProps) {
  const [locale, setLocaleState] = useState(initialLocale);
  const [registry, setRegistry] = useState<Record<string, Record<string, string>>>(messages);

  const setLocale = useCallback((nextLocale: string) => {
    if (!nextLocale || nextLocale === locale) {
      return;
    }
    setLocaleState(nextLocale);
  }, [locale]);

  const loadMessages = useCallback((bundleLocale: string, bundle: Record<string, string>) => {
    setRegistry((prev) => ({
      ...prev,
      [bundleLocale]: {
        ...(prev[bundleLocale] ?? {}),
        ...bundle,
      },
    }));
  }, []);

  const translate = useCallback((key: string, vars?: Record<string, unknown>) => {
    const template = resolveMessage(key, locale, fallbackLocale, registry);
    return formatTemplate(template, vars);
  }, [locale, fallbackLocale, registry]);

  const has = useCallback((key: string) => {
    if (!key) {
      return false;
    }
    const template = resolveMessage(key, locale, fallbackLocale, registry);
    return template !== key;
  }, [locale, fallbackLocale, registry]);

  const value = useMemo<I18nContextValue>(() => ({
    locale,
    fallbackLocale,
    messages: registry,
    setLocale,
    loadMessages,
    translate,
    t: translate,
    has,
  }), [locale, fallbackLocale, registry, setLocale, loadMessages, translate, has]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within an I18nProvider");
  }
  return context;
}

export function useTranslator() {
  const { translate } = useI18n();
  return translate;
}

export function useLocale(): string {
  const { locale } = useI18n();
  return locale;
}

export function useFormatMessage() {
  const { translate } = useI18n();
  return translate;
}
