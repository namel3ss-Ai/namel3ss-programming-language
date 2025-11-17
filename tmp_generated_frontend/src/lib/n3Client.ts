import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  applyPreparedOptimistic,
  prepareOptimisticMutation,
  type OptimisticMutation,
  type PreparedOptimisticMutation,
} from "./optimistic";

export interface DataSourceRef {
  kind: string;
  name: string;
}

export interface TextWidgetConfig {
  id: string;
  type: "text";
  text: string;
  styles?: Record<string, string>;
}

export interface TableWidgetConfig {
  id: string;
  type: "table";
  title: string;
  columns: string[];
  source: DataSourceRef;
}

export interface ChartWidgetConfig {
  id: string;
  type: "chart";
  title: string;
  chartType?: string;
  source: DataSourceRef;
  x?: string | null;
  y?: string | null;
}

export interface FormWidgetField {
  name: string;
  type?: string;
}

export interface FormWidgetConfig {
  id: string;
  type: "form";
  title: string;
  fields: FormWidgetField[];
  successMessage?: string | null;
}

export type WidgetConfig =
  | TextWidgetConfig
  | TableWidgetConfig
  | ChartWidgetConfig
  | FormWidgetConfig;

export interface NavigationMetadata {
  label?: string;
  hideWhenAuthenticated?: boolean;
  onlyWhenAuthenticated?: boolean;
}

export interface PageDefinition {
  slug: string;
  route: string;
  title: string;
  description?: string | null;
  reactive: boolean;
  realtime: boolean;
  widgets: WidgetConfig[];
  preview: Record<string, unknown>;
  requiresAuth?: boolean;
  allowedRoles?: readonly string[];
  public?: boolean;
  redirectTo?: string | null;
  showInNav?: boolean;
  nav?: NavigationMetadata | null;
  isLogin?: boolean;
  skipHydration?: boolean;
}

export interface PageDataState {
  data: Record<string, unknown> | null;
  loading: boolean;
  error: string | null;
  pageErrors: string[];
  fieldErrors: Record<string, string[]>;
}

export interface PageDataReloadOptions {
  silent?: boolean;
}

export interface ApplyRealtimeOptions {
  replace?: boolean;
}

export interface RequestError extends Error {
  status?: number;
  fieldErrors?: Record<string, string[]>;
  pageErrors?: string[];
  raw?: unknown;
}

export interface UsePageDataResult extends PageDataState {
  reload: (options?: PageDataReloadOptions) => Promise<void>;
  applyRealtime: (payload: unknown, options?: ApplyRealtimeOptions) => void;
  applyOptimistic: (mutation: OptimisticMutation<Record<string, unknown>>) => string;
  rollbackOptimistic: (id?: string) => void;
  clearOptimistic: () => void;
}

export interface MergeOptions {
  copy?: boolean;
}

const DEFAULT_TIMEOUT = 15000;

function mergeHeaders(existing: HeadersInit | undefined, overrides: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {};
  if (existing) {
    if (Array.isArray(existing)) {
      for (const entry of existing) {
        if (!entry || entry.length < 2) {
          continue;
        }
        result[String(entry[0])] = String(entry[1]);
      }
    } else if (typeof Headers !== "undefined" && existing instanceof Headers) {
      existing.forEach((value, key) => {
        result[key] = value;
      });
    } else {
      Object.keys(existing as Record<string, string>).forEach((key) => {
        result[key] = String((existing as Record<string, string>)[key]);
      });
    }
  }
  Object.entries(overrides).forEach(([key, value]) => {
    result[key] = value;
  });
  return result;
}

async function requestJson<T = unknown>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT): Promise<T | null> {
  const controller = new AbortController();
  const signals: AbortSignal[] = [];
  if (init?.signal) {
    signals.push(init.signal);
  }
  const combined = combineAbortSignals(controller.signal, signals);
  const timer = typeof window !== "undefined" ? window.setTimeout(() => controller.abort(), timeoutMs) : undefined;

  try {
    const response = await fetch(path, { ...init, signal: combined });
    const contentType = response.headers.get("content-type") ?? "";
    const isJson = contentType.includes("application/json");
    const payload = isJson ? await response.json().catch(() => null) : null;

    if (!response.ok) {
      const error = createRequestError(response.status, payload);
      throw error;
    }
    return payload as T;
  } finally {
    if (typeof window !== "undefined" && timer !== undefined) {
      window.clearTimeout(timer);
    }
  }
}

function combineAbortSignals(primary: AbortSignal, extras: AbortSignal[]): AbortSignal {
  if (!extras.length) {
    return primary;
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  const signals = [primary, ...extras];
  signals.forEach((signal) => {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", abort, { once: true });
    }
  });
  return controller.signal;
}

function createRequestError(status: number, payload: unknown): RequestError {
  const message = resolveErrorMessage(payload) ?? `Request failed with status ${status}`;
  const error: RequestError = new Error(message);
  error.status = status;
  if (payload && typeof payload === "object") {
    const fieldErrors = extractFieldErrors(payload as Record<string, unknown>);
    if (Object.keys(fieldErrors).length) {
      error.fieldErrors = fieldErrors;
    }
    const pageErrors = extractPageErrors(payload as Record<string, unknown>);
    if (pageErrors.length) {
      error.pageErrors = pageErrors;
    }
    error.raw = payload;
  }
  return error;
}

function resolveErrorMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const detail = (payload as Record<string, unknown>).detail ?? (payload as Record<string, unknown>).message;
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (Array.isArray((payload as Record<string, unknown>).errors)) {
    const first = (payload as Record<string, unknown>).errors[0];
    if (first && typeof first === "object" && typeof first.message === "string") {
      return first.message;
    }
  }
  return null;
}

function extractFieldErrors(payload: Record<string, unknown>): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  const candidates = [payload.fieldErrors, payload.field_errors, payload.errors];
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    if (Array.isArray(candidate)) {
      for (const entry of candidate) {
        if (entry && typeof entry === "object" && typeof entry.field === "string") {
          const value = typeof entry.message === "string" ? entry.message : String(entry.detail ?? "");
          if (!result[entry.field]) {
            result[entry.field] = [];
          }
          if (value) {
            result[entry.field].push(value);
          }
        }
      }
    } else if (typeof candidate === "object") {
      Object.entries(candidate as Record<string, unknown>).forEach(([field, value]) => {
        if (Array.isArray(value)) {
          result[field] = value.map((item) => String(item));
        } else if (typeof value === "string") {
          result[field] = [value];
        }
      });
    }
  }
  return result;
}

function extractPageErrors(payload: Record<string, unknown>): string[] {
  const all: string[] = [];
  const candidates = [payload.pageErrors, payload.page_errors, payload.errors];
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    if (Array.isArray(candidate)) {
      candidate.forEach((entry) => {
        if (typeof entry === "string" && entry.trim()) {
          all.push(entry.trim());
        } else if (entry && typeof entry === "object" && typeof (entry as { message?: string }).message === "string") {
          all.push(String((entry as { message?: string }).message));
        }
      });
    }
  }
  return all;
}

export async function fetchResource<T = unknown>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT): Promise<T | null> {
  const headers = mergeHeaders(init?.headers, { Accept: "application/json" });
  return requestJson<T>(path, { ...init, headers }, timeoutMs);
}

export async function submitJson<T = unknown>(path: string, payload?: unknown, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT): Promise<T | null> {
  const headers = mergeHeaders(init?.headers, { Accept: "application/json", "Content-Type": "application/json" });
  const body = init?.body ?? JSON.stringify(payload ?? {});
  const method = init?.method ?? "POST";
  return requestJson<T>(path, { ...init, method, headers, body }, timeoutMs);
}

export function mergePartial<T extends Record<string, unknown>>(
  target: T | null | undefined,
  updates: Record<string, unknown> | null | undefined,
  options?: MergeOptions,
): T {
  const shouldCopy = options?.copy !== false;
  const base: Record<string, unknown> = target && typeof target === "object"
    ? (shouldCopy ? { ...(target as Record<string, unknown>) } : (target as Record<string, unknown>))
    : {};
  if (!updates || typeof updates !== "object") {
    return base as T;
  }
  Object.entries(updates).forEach(([key, value]) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const existing = base[key];
      const nextBase = existing && typeof existing === "object" && !Array.isArray(existing)
        ? (existing as Record<string, unknown>)
        : {};
      base[key] = mergePartial(nextBase, value as Record<string, unknown>, { copy: shouldCopy });
    } else {
      base[key] = value;
    }
  });
  return base as T;
}

export function usePageData(definition: PageDefinition): UsePageDataResult {
  const initialData = definition.skipHydration ? {} : null;
  const [state, setState] = useState<PageDataState>({
    data: initialData,
    loading: !definition.skipHydration,
    error: null,
    pageErrors: [],
    fieldErrors: {},
  });

  const abortRef = useRef<AbortController | null>(null);
  const baseDataRef = useRef<Record<string, unknown> | null>(definition.skipHydration ? {} : null);
  const optimisticRef = useRef<PreparedOptimisticMutation<Record<string, unknown>>[]>([]);
  const mountedRef = useRef(true);

  useEffect(() => () => {
    mountedRef.current = false;
    abortRef.current?.abort();
  }, []);

  const recomputeData = useCallback(
    (authoritative: Record<string, unknown> | null) => {
      const base = authoritative ? { ...authoritative } : {};
      return optimisticRef.current.reduce((draft, mutation) => applyPreparedOptimistic(draft, mutation), base);
    },
    [],
  );

  const setAuthoritative = useCallback((payload: Record<string, unknown> | null, pageErrors?: string[], fieldErrors?: Record<string, string[]>) => {
    baseDataRef.current = payload ? { ...payload } : {};
    setState({
      data: recomputeData(baseDataRef.current),
      loading: false,
      error: null,
      pageErrors: pageErrors ?? [],
      fieldErrors: fieldErrors ?? {},
    });
  }, [recomputeData]);

  const fetchData = useCallback(async (options?: PageDataReloadOptions) => {
    if (definition.skipHydration) {
      setState((prev) => ({ ...prev, loading: false }));
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const silent = Boolean(options?.silent);

    setState((prev) => ({
      data: silent ? prev.data : null,
      loading: silent ? prev.loading : true,
      error: silent ? prev.error : null,
      pageErrors: silent ? prev.pageErrors : [],
      fieldErrors: silent ? prev.fieldErrors : {},
    }));

    try {
      const payload = await fetchResource<Record<string, unknown> | null>(`/api/pages/${definition.slug}`, {
        signal: controller.signal,
      });
      if (controller.signal.aborted || !mountedRef.current) {
        return;
      }
      const pageErrors = Array.isArray(payload?.pageErrors)
        ? (payload?.pageErrors as string[])
        : extractPageErrors((payload ?? {}) as Record<string, unknown>);
      const fieldErrors = extractFieldErrors((payload ?? {}) as Record<string, unknown>);
      const normalised = payload && typeof payload === "object" && "data" in payload
        ? mergePartial({}, (payload as Record<string, unknown>).data as Record<string, unknown>)
        : (payload ?? {});
      setAuthoritative(normalised as Record<string, unknown>, pageErrors, fieldErrors);
    } catch (error) {
      if (controller.signal.aborted || !mountedRef.current) {
        return;
      }
      const requestError = error as RequestError;
      setState((prev) => ({
        data: silent ? prev.data : null,
        loading: false,
        error: requestError.message,
        pageErrors: requestError.pageErrors ?? [],
        fieldErrors: requestError.fieldErrors ?? {},
      }));
    }
  }, [definition.skipHydration, definition.slug, recomputeData, setAuthoritative]);

  useEffect(() => {
    fetchData();
    return () => abortRef.current?.abort();
  }, [fetchData]);

  const reload = useCallback((options?: PageDataReloadOptions) => fetchData(options), [fetchData]);

  const applyRealtime = useCallback((payload: unknown, options?: ApplyRealtimeOptions) => {
    if (!payload || typeof payload !== "object" || !mountedRef.current) {
      return;
    }
    const replace = options?.replace ?? false;
    setState((prev) => {
      const authoritative = replace
        ? mergePartial({}, payload as Record<string, unknown>)
        : mergePartial(baseDataRef.current ?? {}, payload as Record<string, unknown>);
      baseDataRef.current = authoritative;
      return {
        data: recomputeData(authoritative),
        loading: false,
        error: null,
        pageErrors: prev.pageErrors,
        fieldErrors: prev.fieldErrors,
      };
    });
  }, [recomputeData]);

  const applyOptimistic = useCallback((mutation: OptimisticMutation<Record<string, unknown>>) => {
    const prepared = prepareOptimisticMutation<Record<string, unknown>>(mutation);
    optimisticRef.current = [...optimisticRef.current.filter((entry) => entry.id !== prepared.id), prepared];
    setState((prev) => ({
      data: recomputeData(baseDataRef.current),
      loading: prev.loading,
      error: prev.error,
      pageErrors: prev.pageErrors,
      fieldErrors: prev.fieldErrors,
    }));
    return prepared.id;
  }, [recomputeData]);

  const rollbackOptimistic = useCallback((id?: string) => {
    optimisticRef.current = id
      ? optimisticRef.current.filter((entry) => entry.id !== id)
      : [];
    setState((prev) => ({
      data: recomputeData(baseDataRef.current),
      loading: prev.loading,
      error: prev.error,
      pageErrors: prev.pageErrors,
      fieldErrors: prev.fieldErrors,
    }));
  }, [recomputeData]);

  const clearOptimistic = useCallback(() => {
    optimisticRef.current = [];
    setState((prev) => ({
      data: recomputeData(baseDataRef.current),
      loading: prev.loading,
      error: prev.error,
      pageErrors: prev.pageErrors,
      fieldErrors: prev.fieldErrors,
    }));
  }, [recomputeData]);

  return useMemo(() => ({
    ...state,
    reload,
    applyRealtime,
    applyOptimistic,
    rollbackOptimistic,
    clearOptimistic,
  }), [state, reload, applyRealtime, applyOptimistic, rollbackOptimistic, clearOptimistic]);
}

export function resolveWidgetData(widgetId: string, pageData: Record<string, unknown> | null | undefined): unknown {
  if (!pageData || typeof pageData !== "object") {
    return undefined;
  }

  const toRecord = (value: unknown): Record<string, unknown> | null => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return null;
    }
    return value as Record<string, unknown>;
  };

  const buckets = [
    (pageData as Record<string, unknown>).widgets,
    (pageData as Record<string, unknown>).components,
    (pageData as Record<string, unknown>).data,
  ];

  let resolved: unknown;
  for (const bucket of buckets) {
    const record = toRecord(bucket);
    if (record && widgetId in record) {
      resolved = record[widgetId];
      break;
    }
  }

  if (resolved === undefined && widgetId in (pageData as Record<string, unknown>)) {
    resolved = (pageData as Record<string, unknown>)[widgetId];
  }

  const optimisticBucket = toRecord((pageData as Record<string, unknown>).optimistic);
  if (!optimisticBucket) {
    return resolved;
  }

  const optimisticEntry = optimisticBucket[widgetId];
  if (optimisticEntry === undefined) {
    return resolved;
  }

  const optimisticRecord = toRecord(optimisticEntry);
  if (!optimisticRecord) {
    return optimisticEntry ?? resolved;
  }

  const { data: optimisticData, ...meta } = optimisticRecord;
  let next: unknown = resolved;

  if (optimisticData !== undefined) {
    const optimisticDataRecord = toRecord(optimisticData);
    if (optimisticDataRecord) {
      const baseRecord = toRecord(next) ?? {};
      next = mergePartial<Record<string, unknown>>(baseRecord, optimisticDataRecord);
    } else {
      next = optimisticData;
    }
  }

  const metaKeys = Object.keys(meta);
  if (metaKeys.length) {
    const baseRecordForMeta = toRecord(next);
    if (baseRecordForMeta) {
      next = mergePartial<Record<string, unknown>>(baseRecordForMeta, meta as Record<string, unknown>);
    } else if (next === undefined) {
      next = meta;
    } else {
      next = { value: next, ...meta };
    }
  }

  return next;
}

export function ensureArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}
